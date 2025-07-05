"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { toast } from "sonner";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { TodoInput, todoSchema } from "@/lib/schemas";
import ChatBox from "@/components/ChatBox";

export default function Home() {
  const form = useForm<TodoInput>({
    resolver: zodResolver(todoSchema),
    defaultValues: {
      title: "",
      description: "",
    },
  });

  const onSubmit = (data: TodoInput) => {
    toast.success("Todo created!", {
      description: `Title: ${data.title}`,
    });
    form.reset();
  };

  return (
    <main className="container mx-auto p-4 space-y-8">
      <ChatBox />
    </main>
  );
}
