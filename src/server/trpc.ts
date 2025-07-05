import { initTRPC } from "@trpc/server";
import { z } from "zod";
import OpenAI from "openai";

const t = initTRPC.create();

// Base router and procedure helpers
export const router = t.router;
export const publicProcedure = t.procedure;

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Create a router instance
export const appRouter = router({
  chat: publicProcedure
    .input(z.object({ message: z.string() }))
    .mutation(async ({ input }) => {
      try {
        const completion = await openai.chat.completions.create({
          messages: [
            {
              role: "system",
              content:
                "You are a helpful assistant. Provide clear and concise responses.",
            },
            {
              role: "user",
              content: input.message,
            },
          ],
          model: "gpt-3.5-turbo",
          temperature: 0.7,
          max_tokens: 150,
        });

        return {
          response:
            completion.choices[0]?.message?.content ||
            "I couldn't generate a response.",
        };
      } catch (error) {
        console.error("OpenAI API Error:", error);
        return {
          response:
            "Sorry, I encountered an error while processing your request.",
        };
      }
    }),
});

// Export type router type signature
export type AppRouter = typeof appRouter;
