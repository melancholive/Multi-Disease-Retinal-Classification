import { spawn } from 'child_process';
import { writeFileSync, mkdirSync, readdirSync, statSync } from 'fs';
import { join } from 'path';
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest): Promise<Response> {
  try {
    const { image } = await request.json();

    if (!image) {
      return NextResponse.json(
        { error: 'No image provided' },
        { status: 400 }
      );
    }

    // Create .tmp directory if it doesn't exist
    const tmpDir = join(process.cwd(), '.tmp');
    try {
      mkdirSync(tmpDir, { recursive: true });
    } catch {
      // Directory might already exist
    }

    // Save base64 image data to temporary file
    const timestamp = Date.now();
    const imageFile = join(tmpDir, `img_${timestamp}.txt`);
    writeFileSync(imageFile, image);

    // Get the model path
    const modelPath = join(process.cwd(), 'models', 'resnet18-weights.pth');

    // Run Python inference script
    const pythonProcess = spawn('python3', [
      join(process.cwd(), 'lib', 'model.py'),
      imageFile,
      modelPath,
    ]);

    let output = '';
    let error = '';

    return new Promise((resolve) => {
      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        error += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          console.error('Python error:', error);
          resolve(
            NextResponse.json(
              { error: 'Failed to process image', details: error },
              { status: 500 }
            )
          );
          return;
        }

        try {
          const predictions = JSON.parse(output);

          // Find the most recently saved preprocessed image
          let preprocessedImage: string | undefined;
          try {
            const files = readdirSync(tmpDir);
            const preprocessedFiles = files
              .filter((f) => f.startsWith('preprocessed_') && f.endsWith('.png'))
              .map((f) => ({
                name: f,
                time: statSync(join(tmpDir, f)).mtimeMs,
              }))
              .sort((a, b) => b.time - a.time);

            if (preprocessedFiles.length > 0) {
              const latestFile = preprocessedFiles[0].name;
              const timestampMatch = latestFile.match(/preprocessed_(\d+)\.png/);
              if (timestampMatch) {
                preprocessedImage = `/api/preprocessed/${timestampMatch[1]}`;
              }
            }
          } catch (err) {
            console.warn('Could not find preprocessed image:', err);
          }

          const response = {
            predictions,
            ...(preprocessedImage && { preprocessedImage }),
          };

          resolve(NextResponse.json(response, { status: 200 }));
        } catch (err) {
          console.error('Parse error:', err, 'Output:', output);
          resolve(
            NextResponse.json(
              { error: 'Failed to parse predictions' },
              { status: 500 }
            )
          );
        }
      });
    });
  } catch (error) {
    console.error('Error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
