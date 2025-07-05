import { OpenAI } from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(req: Request) {
  const { message } = await req.json();

  const encoder = new TextEncoder();
  const stream = new TransformStream();
  const writer = stream.writable.getWriter();

  const writeStream = async (text: string) => {
    await writer.write(encoder.encode(`data: ${JSON.stringify({ text })}\n\n`));
  };

  // Start streaming response
  streamResponse(message, writeStream, writer);

  return new Response(stream.readable, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}

async function streamResponse(
  message: string,
  writeStream: (text: string) => Promise<void>,
  writer: WritableStreamDefaultWriter<Uint8Array>
) {
  try {
    const stream = await openai.chat.completions.create({
      messages: [
        {
          role: "system",
          content:
            "You are a helpful assistant. Provide clear and concise responses.",
        },
        {
          role: "user",
          content: message,
        },
      ],
      model: "gpt-3.5-turbo",
      stream: true,
    });

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || "";
      if (content) {
        await writeStream(content);
      }
    }
  } catch (error) {
    console.error("OpenAI API Error:", error);
    await writeStream("[ERROR] Failed to generate response.");
  } finally {
    await writer.close();
  }
}
