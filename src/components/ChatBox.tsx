"use client";

import { useState, useEffect, useRef } from "react";
import { Button } from "./ui/button";
import { Card } from "./ui/card";
import { Input } from "./ui/input";
import { ScrollArea } from "./ui/scroll-area";

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface AgentEvent {
  type:
    | "thinking"
    | "tool_call"
    | "tool_result"
    | "message_chunk"
    | "complete"
    | "error"
    | "reasoning"
    | "handoff"
    | "handoff_complete"
    | "mcp"
    | "unknown_run_item"
    | "agent_lifecycle"
    | "handoff_lifecycle"
    | "tool_lifecycle";
  data: any;
  timestamp: number;
}

interface AssistantMessage extends Message {
  role: "assistant";
  events: AgentEvent[];
  isStreaming: boolean;
}

export default function ChatBox() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState<(Message | AssistantMessage)[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!message.trim()) return;

    const userMessage: Message = { role: "user", content: message };
    setMessages((prev) => [...prev, userMessage]);

    const currentMessage = message;
    setMessage("");
    setIsLoading(true);

    try {
      // First, send the message via POST
      const response = await fetch("http://localhost:8000/chat/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: currentMessage }),
        credentials: "include",
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Create initial assistant message
      const assistantMessage: AssistantMessage = {
        role: "assistant",
        content: "",
        events: [],
        isStreaming: true,
      };
      setMessages((prev) => [...prev, assistantMessage]);

      // Read the streaming response
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("Failed to get response reader");
      }

      const decoder = new TextDecoder();
      let buffer = "";
      let currentEventType = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");

        // Keep the last incomplete line in buffer
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEventType = line.slice(7).trim();
            continue;
          }

          if (line.startsWith("data: ")) {
            const dataStr = line.slice(6);
            if (dataStr.trim() === "") continue;

            try {
              const eventData = JSON.parse(dataStr);
              const event: AgentEvent = {
                type: currentEventType as any,
                data: eventData,
                timestamp: Date.now(),
              };

              setMessages((prev) => {
                const newMessages = [...prev];
                const lastMessage = newMessages[
                  newMessages.length - 1
                ] as AssistantMessage;

                if (lastMessage.role === "assistant") {
                  const updatedMessage = { ...lastMessage };
                  updatedMessage.events = [...updatedMessage.events, event];

                  // Handle different event types
                  if (event.type === "message_chunk" && event.data.word) {
                    updatedMessage.content += event.data.word + " ";
                  } else if (event.type === "complete") {
                    updatedMessage.isStreaming = false;
                  }

                  newMessages[newMessages.length - 1] = updatedMessage;
                }

                return newMessages;
              });
            } catch (error) {
              console.error("Failed to parse data:", dataStr);
            }
          }
        }
      }
    } catch (error) {
      const errorMessage: AssistantMessage = {
        role: "assistant",
        content: "Error occurred while processing your request",
        events: [
          {
            type: "error",
            data: {
              error: error instanceof Error ? error.message : "Unknown error",
            },
            timestamp: Date.now(),
          },
        ],
        isStreaming: false,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    messages.forEach((msg, index) => {
      if (msg.role === "assistant") {
        const assistantMsg = msg as AssistantMessage;

        const lifecycleEvents = assistantMsg.events.filter(
          (event) =>
            event.type === "agent_lifecycle" || event.type === "tool_lifecycle"
        );
      }
    });
  }, [messages]);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }

    console.log("🔍 MESSAGES:", messages);
  }, [messages]);

  return (
    <Card className="w-full max-w-2xl mx-auto h-[600px] flex flex-col shadow-none border-none">
      <div className="flex-1 p-4 max-h-[calc(100vh-80px)]">
        <ScrollArea ref={scrollAreaRef} className="h-full pr-4">
          <div className="space-y-4">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`flex ${
                  msg.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[80%] rounded-lg p-3 ${
                    msg.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted"
                  }`}
                >
                  {msg.role === "user" ? (
                    <p className="whitespace-pre-wrap break-words">
                      {msg.content}
                    </p>
                  ) : (
                    <div className="space-y-2">
                      {/* Render lifecycle events only */}
                      {(() => {
                        const lifecycleEvents = (
                          msg as AssistantMessage
                        ).events.filter(
                          (event) =>
                            event.type === "agent_lifecycle" ||
                            event.type === "tool_lifecycle"
                        );

                        return lifecycleEvents.map((event, eventIndex) => {
                          const isLastEvent =
                            eventIndex === lifecycleEvents.length - 1;
                          const isStreaming = (msg as AssistantMessage)
                            .isStreaming;

                          return (
                            <div key={eventIndex}>
                              {event.type === "agent_lifecycle" && (
                                <>
                                  {event.data.output && (
                                    <div className="text-sm text-gray-700 leading-relaxed">
                                      {event.data.output}
                                    </div>
                                  )}
                                </>
                              )}

                              {event.type === "tool_lifecycle" && (
                                <div className="flex flex-col gap-2">
                                  <div className="space-y-3">
                                    {event.data.context === "tool_start" && (
                                      <div className="flex items-center gap-2">
                                        <div className="flex-shrink-0">
                                          {isLastEvent && isStreaming ? (
                                            <div className="animate-spin w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
                                          ) : (
                                            <span className="text-teal-700 font-medium pl-2">
                                              {event.data.tool_name !==
                                                "reflect_on_progress" &&
                                                event.data.tool_name !==
                                                  "observe_result" &&
                                                "Ran tool"}{" "}
                                            </span>
                                          )}
                                        </div>

                                        <div className="text-sm font-medium text-gray-900">
                                          {" "}
                                          {event.data.tool_name ===
                                          "observe_result"
                                            ? "Making Observation"
                                            : event.data.tool_name ===
                                              "reflect_on_progress"
                                            ? "Reflecting on Progress"
                                            : event.data.tool_name}
                                        </div>
                                      </div>
                                    )}

                                    {event.data.result && (
                                      <div className="flex items-center gap-2">
                                        <div className="text-sm text-gray-700 leading-relaxed flex items-center gap-2 pl-2">
                                          <span>
                                            {event.data.tool_name !==
                                              "observe_result" &&
                                              event.data.tool_name !==
                                                "reflect_on_progress" &&
                                              "Result:"}
                                          </span>
                                          <span
                                            className={
                                              event.data.tool_name ===
                                              "observe_result"
                                                ? "italic text-blue-700"
                                                : event.data.tool_name ===
                                                  "reflect_on_progress"
                                                ? "italic text-purple-700"
                                                : ""
                                            }
                                          >
                                            {event.data.result}
                                          </span>
                                        </div>
                                      </div>
                                    )}
                                  </div>
                                </div>
                              )}
                            </div>
                          );
                        });
                      })()}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </div>

      <div className="p-4 fixed bottom-0 w-2xl">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <Input
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 bg-white"
          />
          <Button type="submit" disabled={isLoading}>
            {isLoading ? "..." : "Send"}
          </Button>
        </form>
      </div>
    </Card>
  );
}
