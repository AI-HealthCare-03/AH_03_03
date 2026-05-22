import { apiRequest } from "./client";

export type ChatbotContextType = "MAIN" | "ANALYSIS" | "DIET" | "CHALLENGE" | "GENERAL";

export type ChatbotAskRequest = {
  message: string;
  context_type?: ChatbotContextType;
  target_id?: number | null;
};

export type ChatbotAskResponse = {
  answer: string;
  source: "DUMMY_LLM" | string;
  context_type: ChatbotContextType;
  recommended_actions: string[];
  safety_notice: string;
};

export async function askChatbot(payload: ChatbotAskRequest): Promise<ChatbotAskResponse> {
  return apiRequest<ChatbotAskResponse>("/chatbot/ask", {
    method: "POST",
    body: payload,
  });
}
