"use client";

import { useState, useEffect } from "react";
import { Button } from "./ui/button";
import { Card } from "./ui/card";
import { Input } from "./ui/input";
import { ScrollArea } from "./ui/scroll-area";
import {
  fetchEventSource,
  EventSourceMessage,
} from "@microsoft/fetch-event-source";
import ReactMarkdown from "react-markdown";

interface Message {
  role: "user" | "assistant";
  content: string;
}

type StreamEvent = any;

interface AssistantMessage extends Message {
  role: "assistant";
  events: StreamEvent[];
  isStreaming?: boolean;
}

export default function ChatBox() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState<(Message | AssistantMessage)[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [functionCallId, setFunctionCallId] = useState<string | null>(null);
  const [dots, setDots] = useState(".");

  useEffect(() => {
    if (!isLoading) return;
    let current = 0;
    const dotArr = [".", "..", "..."];
    const interval = setInterval(() => {
      current = (current + 1) % dotArr.length;
      setDots(dotArr[current]);
    }, 250);
    return () => clearInterval(interval);
  }, [isLoading]);

  useEffect(() => {
    let foundFunctionCall = false;
    let foundFinished = false;
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (msg.role === "assistant" && (msg as AssistantMessage).events) {
        const events = (msg as AssistantMessage).events;
        for (let j = events.length - 1; j >= 0; j--) {
          const e = events[j];
          if (e.status === "finished") {
            foundFinished = true;
            break;
          }
          if (
            e.data &&
            e.data.item &&
            e.data.item.type === "function_call" &&
            e.data.item.status === "completed"
          ) {
            setFunctionCallId(e.data.item.call_id);
            foundFunctionCall = true;
            break;
          }
        }
      }
      if (foundFunctionCall || foundFinished) break;
    }
    if (foundFinished) {
      setFunctionCallId(null);
      setIsLoading(false);
      // Only update messages if isStreaming is actually true
      setMessages((prev) => {
        const newMessages = [...prev];
        for (let i = newMessages.length - 1; i >= 0; i--) {
          const msg = newMessages[i];
          if (
            msg.role === "assistant" &&
            (msg as AssistantMessage).isStreaming
          ) {
            if ((msg as AssistantMessage).isStreaming) {
              newMessages[i] = { ...msg, isStreaming: false };
              return newMessages; // Only update if changed
            }
            break;
          }
        }
        return prev; // No change, return previous state
      });
    } else if (!foundFunctionCall) {
      setFunctionCallId(null);
    }
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!message.trim()) return;

    const userMessage: Message = { role: "user", content: message };
    setMessages((prev) => [...prev, userMessage]);

    const currentMessage = message;
    setMessage("");
    setIsLoading(true);

    // Create initial assistant message for streaming
    const assistantMessage: AssistantMessage = {
      role: "assistant",
      content: "",
      events: [],
      isStreaming: true,
    };
    setMessages((prev) => [...prev, assistantMessage]);

    try {
      await fetchEventSource("http://localhost:8000/chat/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({ message: currentMessage }),
        async onopen(res: Response) {
          if (res.ok && res.status === 200) {
            console.log("Connection established", res);
          } else if (
            res.status >= 400 &&
            res.status < 500 &&
            res.status !== 429
          ) {
            throw new Error(`HTTP error! status: ${res.status}`);
          }
        },
        onmessage(event: EventSourceMessage) {
          try {
            // console.log("event", event);

            const eventType = event.event;
            const dataStr = event.data;
            let eventData: StreamEvent | null = null;

            try {
              const parsedData = JSON.parse(dataStr);
              eventData = {
                ...parsedData,
              };
              console.log("eventData", eventData);
            } catch {
              const deltaMatch = dataStr.match(/delta='(.*?)'/);
              if (deltaMatch) {
                eventData = {
                  type: eventType,
                  data: { delta: deltaMatch[1] },
                };
              }
            }

            if (eventData) {
              setMessages((prev) => {
                const newMessages = [...prev];
                const lastMessage = newMessages[
                  newMessages.length - 1
                ] as AssistantMessage;

                if (lastMessage.role === "assistant") {
                  const updatedMessage = { ...lastMessage };
                  updatedMessage.events = [
                    ...updatedMessage.events,
                    eventData!,
                  ];
                  newMessages[newMessages.length - 1] = updatedMessage;
                }

                return newMessages;
              });
            }
          } catch (error) {
            console.error("Failed to parse event:", error);
          }
        },
        onclose() {
          console.log("Connection closed by the server");
          setIsLoading(false);
        },
        onerror(err: Error) {
          console.error("Streaming error:", err);
          setMessages((prev) => {
            const newMessages = [...prev];
            const errorMessage: Message = {
              role: "assistant",
              content: "Sorry, there was an error processing your request.",
            };

            if (
              newMessages[newMessages.length - 1]?.role === "assistant" &&
              (newMessages[newMessages.length - 1] as AssistantMessage)
                .content === ""
            ) {
              newMessages[newMessages.length - 1] = errorMessage;
            } else {
              newMessages.push(errorMessage);
            }
            return newMessages;
          });
          setIsLoading(false);
        },
      });
    } catch (error) {
      console.error("Connection error:", error);
      setMessages((prev) => {
        const newMessages = [...prev];
        const errorMessage: Message = {
          role: "assistant",
          content: "Sorry, there was an error connecting to the server.",
        };

        if (
          newMessages[newMessages.length - 1]?.role === "assistant" &&
          (newMessages[newMessages.length - 1] as AssistantMessage).content ===
            ""
        ) {
          newMessages[newMessages.length - 1] = errorMessage;
        } else {
          newMessages.push(errorMessage);
        }
        return newMessages;
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex justify-center items-center h-screen min-h-screen bg-background">
      <div className="w-full max-w-lg h-full flex flex-col shadow-md border-none bg-background">
        <div className="flex-1 p-4 pb-2 overflow-hidden">
          <ScrollArea className="h-full pr-2">
            <div className="space-y-3">
              {messages.map((msg, index) => (
                <div
                  key={index}
                  className={`flex ${
                    msg.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-[75%] px-3 py-2 text-sm shadow-none border border-transparent ${
                      msg.role === "user"
                        ? "bg-primary text-primary-foreground rounded-br-md rounded-tl-xl rounded-bl-xl"
                        : "bg-muted text-muted-foreground rounded-bl-md rounded-tr-xl rounded-br-xl"
                    }`}
                  >
                    {msg.role === "assistant" ? (
                      <div className="whitespace-pre-wrap">
                        {(() => {
                          // Build output chunks: buffer markdown, interleave tool components
                          const outputChunks: Array<
                            | { type: "markdown"; content: string }
                            | { type: "tool"; element: React.ReactNode }
                          > = [];
                          let currentMarkdown = "";
                          (msg as AssistantMessage).events.forEach(
                            (e, eventIdx) => {
                              if (
                                e.data &&
                                e.data.type === "response.output_text.delta" &&
                                typeof e.data.delta === "string"
                              ) {
                                currentMarkdown += e.data.delta;
                              } else if (
                                e.data &&
                                e.data.item &&
                                e.data.item.type === "function_call" &&
                                e.data.item.status === "completed"
                              ) {
                                // Flush buffered markdown before tool
                                if (currentMarkdown) {
                                  outputChunks.push({
                                    type: "markdown",
                                    content: currentMarkdown,
                                  });
                                  currentMarkdown = "";
                                }
                                outputChunks.push({
                                  type: "tool",
                                  element: (
                                    <div
                                      key={eventIdx}
                                      className="mt-2 flex flex-row items-center gap-2 px-2 py-1 w-fit bg-muted text-muted-foreground rounded"
                                    >
                                      {e.data.item.call_id ===
                                        functionCallId && (
                                        <span className="inline-block w-3 h-3 bg-muted-foreground/40 animate-spin rounded-full" />
                                      )}
                                      <span className="text-xs italic">
                                        {e.data.item.name}
                                      </span>
                                    </div>
                                  ),
                                });
                              }
                            }
                          );
                          // Flush any remaining markdown
                          if (currentMarkdown) {
                            outputChunks.push({
                              type: "markdown",
                              content: currentMarkdown,
                            });
                          }
                          // Render the chunks
                          return outputChunks.map((chunk, idx) =>
                            chunk.type === "markdown" ? (
                              <ReactMarkdown
                                key={idx}
                                components={{
                                  p: ({ children }) => <>{children}</>,
                                }}
                              >
                                {chunk.content}
                              </ReactMarkdown>
                            ) : (
                              chunk.element
                            )
                          );
                        })()}
                      </div>
                    ) : (
                      <span className="whitespace-pre-wrap">{msg.content}</span>
                    )}
                  </div>
                </div>
              ))}
              {isLoading && messages.length === 0 && (
                <div className="flex justify-start">
                  <div className="bg-muted text-muted-foreground rounded-xl px-4 py-3 shadow-none border-none w-20 h-6 animate-pulse" />
                </div>
              )}
            </div>
          </ScrollArea>
        </div>
        <form
          onSubmit={handleSubmit}
          className="flex gap-2 p-3 border-t border-muted-foreground/10 bg-background sticky bottom-0"
        >
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 resize-none rounded-md border border-muted-foreground/20 px-3 py-2 text-sm focus-visible:ring-2 focus-visible:ring-primary/50 bg-background min-h-[40px] max-h-32"
            disabled={isLoading}
            autoComplete="off"
            rows={1}
          />
          <Button
            type="submit"
            disabled={isLoading}
            className="h-9 px-5 self-end"
          >
            Send
          </Button>
        </form>
        {isLoading && (
          <div className="flex flex-row items-center mb-2 px-3 py-1 w-fit text-xs text-muted-foreground">
            <span>{`Generating${dots}`}</span>
          </div>
        )}
      </div>
    </div>
  );
}
