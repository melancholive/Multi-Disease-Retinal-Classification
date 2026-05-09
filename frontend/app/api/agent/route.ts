import { NextRequest, NextResponse } from 'next/server';
import http from 'node:http';
import https from 'node:https';

export const runtime = 'nodejs';

const AGENT_PROXY_TIMEOUT_MS = 10 * 60 * 1000;

interface UpstreamResponse {
  body: string;
  contentType: string;
  status: number;
}

function postJson(url: string, payload: unknown): Promise<UpstreamResponse> {
  return new Promise((resolve, reject) => {
    const target = new URL(url);
    const body = JSON.stringify(payload);
    const transport = target.protocol === 'https:' ? https : http;

    const req = transport.request(
      {
        hostname: target.hostname,
        port: target.port,
        path: `${target.pathname}${target.search}`,
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'content-length': Buffer.byteLength(body),
        },
        timeout: AGENT_PROXY_TIMEOUT_MS,
      },
      (res) => {
        const chunks: Buffer[] = [];

        res.on('data', (chunk: Buffer) => {
          chunks.push(chunk);
        });

        res.on('end', () => {
          resolve({
            body: Buffer.concat(chunks).toString('utf8'),
            contentType: res.headers['content-type']?.toString() || 'application/json',
            status: res.statusCode || 502,
          });
        });
      }
    );

    req.on('timeout', () => {
      req.destroy(new Error('Agent request timed out'));
    });
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

export async function POST(request: NextRequest): Promise<Response> {
  try {
    const apiBase = process.env.BACKEND_URL ?? 'http://127.0.0.1:8000';
    const payload = await request.json();

    const upstream = await postJson(`${apiBase}/agent`, payload);

    return new NextResponse(upstream.body, {
      status: upstream.status,
      headers: {
        'content-type': upstream.contentType,
      },
    });
  } catch (error) {
    console.error('Error:', error);
    const message = error instanceof Error ? error.message : 'Internal server error';
    const status = message.includes('timed out') ? 504 : 500;

    return NextResponse.json({ error: message }, { status });
  }
}
