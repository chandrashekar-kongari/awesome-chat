"use client";

import { useState, useEffect, useRef } from "react";
import { Button } from "./ui/button";
import { Card } from "./ui/card";
import { Input } from "./ui/input";
import { ScrollArea } from "./ui/scroll-area";
import katex from "katex";
import "katex/dist/katex.min.css";
import {
  Document,
  Page,
  Text,
  View,
  StyleSheet,
  PDFDownloadLink,
} from "@react-pdf/renderer";

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

interface UploadResult {
  status: string;
  message: string;
  file_path: string;
  size: number;
  instructions: string;
}

interface UploadState {
  isUploading: boolean;
  uploadProgress: number;
  uploadResult: UploadResult | null;
  uploadError: string | null;
}

// File Upload Component
const FileUploadArea = ({
  onFileUpload,
  uploadState,
}: {
  onFileUpload: (file: File) => void;
  uploadState: UploadState;
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      onFileUpload(files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onFileUpload(files[0]);
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <Card className="w-full max-w-2xl mx-auto mb-4 p-4 bg-gradient-to-r from-green-50 to-blue-50 border-2 border-green-200">
      <div className="text-center space-y-4">
        <h3 className="text-lg font-semibold text-gray-800">
          📄 Document Upload & Indexing
        </h3>

        <div
          className={`relative border-2 border-dashed rounded-lg p-8 transition-all duration-200 cursor-pointer ${
            isDragOver
              ? "border-blue-500 bg-blue-50"
              : "border-gray-300 bg-white hover:border-gray-400 hover:bg-gray-50"
          } ${uploadState.isUploading ? "pointer-events-none opacity-50" : ""}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={handleClick}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt,.doc,.docx,.md"
            onChange={handleFileSelect}
            className="hidden"
            disabled={uploadState.isUploading}
          />

          <div className="text-center">
            {uploadState.isUploading ? (
              <div className="space-y-2">
                <div className="text-2xl">⏳</div>
                <div className="text-sm text-gray-600">
                  Uploading and indexing to Pinecone...
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${uploadState.uploadProgress}%` }}
                  />
                </div>
                <div className="text-xs text-gray-500">
                  {uploadState.uploadProgress}%
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                <div className="text-4xl">📎</div>
                <div className="text-sm text-gray-600">
                  Drag & drop your document here or click to select
                </div>
                <div className="text-xs text-gray-500">
                  Supports PDF, TXT, DOC, DOCX, MD files
                </div>
              </div>
            )}
          </div>
        </div>

        {uploadState.uploadResult && (
          <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded">
            <div className="font-semibold">✅ Upload Successful!</div>
            <div className="text-sm mt-1">
              {uploadState.uploadResult.message}
            </div>
            <div className="text-xs mt-1">
              Size: {(uploadState.uploadResult.size / 1024).toFixed(1)} KB
            </div>
          </div>
        )}

        {uploadState.uploadError && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            <div className="font-semibold">❌ Upload Failed</div>
            <div className="text-sm mt-1">{uploadState.uploadError}</div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default function ChatBox() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState<(Message | AssistantMessage)[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  // Upload state
  const [uploadState, setUploadState] = useState<UploadState>({
    isUploading: false,
    uploadProgress: 0,
    uploadResult: null,
    uploadError: null,
  });

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

  // File upload handler
  const handleFileUpload = async (file: File) => {
    setUploadState({
      isUploading: true,
      uploadProgress: 0,
      uploadResult: null,
      uploadError: null,
    });

    try {
      // Create FormData for file upload
      const formData = new FormData();
      formData.append("file", file);

      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadState((prev) => ({
          ...prev,
          uploadProgress: Math.min(prev.uploadProgress + 10, 90),
        }));
      }, 200);

      // Upload file to backend
      const uploadResponse = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });

      clearInterval(progressInterval);

      if (!uploadResponse.ok) {
        throw new Error(`Upload failed: ${uploadResponse.status}`);
      }

      const uploadResult: UploadResult = await uploadResponse.json();

      setUploadState((prev) => ({
        ...prev,
        uploadProgress: 100,
        uploadResult,
      }));

      // Auto-index the document after upload
      await autoIndexDocument(uploadResult.file_path, file.name);
    } catch (error) {
      setUploadState((prev) => ({
        ...prev,
        isUploading: false,
        uploadError: error instanceof Error ? error.message : "Unknown error",
      }));
    }
  };

  // Auto-index document after upload
  const autoIndexDocument = async (filePath: string, fileName: string) => {
    try {
      // Add a system message about indexing
      const indexingMessage: AssistantMessage = {
        role: "assistant",
        content: `🔄 Automatically indexing "${fileName}" to Pinecone...`,
        events: [],
        isStreaming: true,
      };
      setMessages((prev) => [...prev, indexingMessage]);

      // Send indexing command to the chat
      const indexCommand = `Index the document ${filePath} in collection "${fileName.replace(
        /\.[^/.]+$/,
        ""
      )}"`;

      const response = await fetch("http://localhost:8000/chat/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: indexCommand }),
        credentials: "include",
      });

      if (!response.ok) {
        throw new Error(`Indexing failed: ${response.status}`);
      }

      // Process the streaming response for indexing
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("Failed to get indexing response reader");
      }

      const decoder = new TextDecoder();
      let buffer = "";
      let currentEventType = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
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

                  if (event.type === "complete") {
                    updatedMessage.isStreaming = false;
                  }

                  newMessages[newMessages.length - 1] = updatedMessage;
                }

                return newMessages;
              });
            } catch (error) {
              console.error("Failed to parse indexing data:", dataStr);
            }
          }
        }
      }

      // Mark upload as complete
      setUploadState((prev) => ({
        ...prev,
        isUploading: false,
      }));
    } catch (error) {
      console.error("Auto-indexing error:", error);

      // Add error message
      const errorMessage: AssistantMessage = {
        role: "assistant",
        content: `❌ Failed to auto-index "${fileName}": ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
        events: [],
        isStreaming: false,
      };
      setMessages((prev) => [...prev, errorMessage]);

      setUploadState((prev) => ({
        ...prev,
        isUploading: false,
        uploadError: `Indexing failed: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
      }));
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
    <div className="w-full max-w-2xl mx-auto space-y-4">
      {/* File Upload Area */}
      <FileUploadArea
        onFileUpload={handleFileUpload}
        uploadState={uploadState}
      />

      {/* Chat Box */}
      <Card className="h-[600px] flex flex-col shadow-none border-none">
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
                      <div className="whitespace-pre-wrap break-words">
                        {msg.content}
                      </div>
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
          <div className="flex gap-2 mb-2"></div>
          <form onSubmit={handleSubmit} className="flex gap-2">
            <Input
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Type your message... (Use $...$ for inline math, $$...$$ for block math)"
              className="flex-1 bg-white"
            />
            <Button type="submit" disabled={isLoading}>
              {isLoading ? "..." : "Send"}
            </Button>
          </form>
        </div>
      </Card>
    </div>
  );
}
