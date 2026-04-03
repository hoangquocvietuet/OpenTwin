// Proxy import requests to the backend, bypassing Next.js body size limits.
// Streams the response back so SSE progress events reach the client.

const BACKEND = "http://localhost:7860";

export async function POST(request: Request) {
  const body = await request.arrayBuffer();
  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);

  const res = await fetch(`${BACKEND}/api/v2/import`, {
    method: "POST",
    headers,
    body,
  });

  // Stream the SSE response back
  return new Response(res.body, {
    status: res.status,
    headers: {
      "content-type": res.headers.get("content-type") || "text/event-stream",
    },
  });
}
