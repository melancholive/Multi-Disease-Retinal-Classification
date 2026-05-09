import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest): Promise<Response> {
  try {
    const apiBase = process.env.BACKEND_URL ?? 'http://127.0.0.1:8000';
    const incomingForm = await request.formData();

    const image = incomingForm.get('image');
    if (!(image instanceof File)) {
      return NextResponse.json({ error: 'No image provided' }, { status: 400 });
    }

    const topKValue = incomingForm.get('top_k');
    const topK = typeof topKValue === 'string' && topKValue ? topKValue : '5';

    const outgoingForm = new FormData();
    outgoingForm.set('image', image, image.name || 'image');

    const upstream = await fetch(`${apiBase}/predict?top_k=${encodeURIComponent(topK)}` , {
      method: 'POST',
      body: outgoingForm,
    });

    const text = await upstream.text();
    const contentType = upstream.headers.get('content-type') || 'application/json';

    return new NextResponse(text, {
      status: upstream.status,
      headers: {
        'content-type': contentType,
      },
    });
  } catch (error) {
    console.error('Error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
