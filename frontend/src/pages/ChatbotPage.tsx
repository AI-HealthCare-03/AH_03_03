import { useState } from "react";
import { Link } from "react-router-dom";

import { askChatbot, type ChatbotAskResponse, type ChatbotContextType } from "../api/chatbot";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

type ChatMessage = {
  id: number;
  role: "user" | "assistant";
  text: string;
  response?: ChatbotAskResponse;
};

type ContextOption = {
  label: string;
  value: ChatbotContextType;
  promptHint?: string;
};

const contextOptions: ContextOption[] = [
  { label: "일반", value: "GENERAL" },
  { label: "혈당", value: "ANALYSIS", promptHint: "혈당" },
  { label: "혈압", value: "ANALYSIS", promptHint: "혈압" },
  { label: "식단", value: "DIET" },
  { label: "운동", value: "CHALLENGE", promptHint: "운동" },
  { label: "복약", value: "GENERAL", promptHint: "복약" },
];

const exampleQuestions = [
  "공복혈당이 높게 나왔는데 어떻게 관리해야 하나요?",
  "혈압이 높을 때 오늘 할 수 있는 행동은?",
  "저녁 식단 추천해줘",
  "복약 시간을 자주 잊어버려요",
];

function toUserFacingText(text: string) {
  return text;
}

export default function ChatbotPage() {
  const [selectedContext, setSelectedContext] = useState<ContextOption>(contextOptions[0]);
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [error, setError] = useState("");
  const [isSending, setIsSending] = useState(false);

  const sendMessage = async (nextMessage = message) => {
    const trimmedMessage = nextMessage.trim();
    if (!trimmedMessage || isSending) {
      return;
    }

    setError("");
    setIsSending(true);
    const userMessage: ChatMessage = {
      id: Date.now(),
      role: "user",
      text: trimmedMessage,
    };
    setMessages((prev) => [...prev, userMessage]);
    setMessage("");

    try {
      const requestMessage = selectedContext.promptHint
        ? `${selectedContext.promptHint}: ${trimmedMessage}`
        : trimmedMessage;
      const response = await askChatbot({
        message: requestMessage,
        context_type: selectedContext.value,
      });
      const userFacingResponse: ChatbotAskResponse = {
        ...response,
        answer: toUserFacingText(response.answer),
        recommended_actions: (response.recommended_actions ?? []).map(toUserFacingText),
        safety_notice: toUserFacingText(response.safety_notice),
      };
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: "assistant",
          text: userFacingResponse.answer,
          response: userFacingResponse,
        },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "AI 건강 상담 응답을 불러오지 못했습니다.");
    } finally {
      setIsSending(false);
    }
  };

  return (
    <div className="page-stack">
      <div className="page-header">
        <div>
          <h1>AI 건강 상담</h1>
          <p>건강 분석 결과, 식단, 운동, 복약 관련 질문을 해보세요.</p>
        </div>
        <Link className="button secondary" to="/inquiries">
          1:1 문의로 이동
        </Link>
      </div>

      <Card title="상담 주제">
        <div className="chat-context-tabs">
          {contextOptions.map((option) => (
            <button
              className={selectedContext.label === option.label ? "filter-tab active" : "filter-tab"}
              key={option.label}
              onClick={() => setSelectedContext(option)}
              type="button"
            >
              {option.label}
            </button>
          ))}
        </div>
      </Card>

      {error && <ErrorMessage message={error} />}

      <Card title="대화">
        <div className="chat-window">
          {messages.length === 0 && (
            <div className="chat-empty">
              <strong>무엇을 물어볼까요?</strong>
              <p>본 서비스는 진단/처방이 아닌 건강관리 참고 정보를 제공합니다.</p>
              <div className="chat-example-grid">
                {exampleQuestions.map((question) => (
                  <button
                    className="chat-example-card"
                    key={question}
                    onClick={() => void sendMessage(question)}
                    type="button"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((chatMessage) => (
            <div className={`chat-message ${chatMessage.role}`} key={chatMessage.id}>
              <div className="chat-bubble">
                <span className="chat-role">{chatMessage.role === "user" ? "나" : "AI 건강 상담"}</span>
                <p>{chatMessage.text}</p>
                {chatMessage.response?.recommended_actions && chatMessage.response.recommended_actions.length > 0 && (
                  <div className="chip-list">
                    {chatMessage.response.recommended_actions.map((action) => (
                      <span className="badge badge-reference" key={action}>
                        {action}
                      </span>
                    ))}
                  </div>
                )}
                {chatMessage.response?.safety_notice && (
                  <p className="chat-safety-notice">{chatMessage.response.safety_notice}</p>
                )}
              </div>
            </div>
          ))}

          {isSending && <div className="state-box">AI 건강 상담이 답변을 준비하고 있습니다.</div>}
        </div>

        <form
          className="chat-input-row"
          onSubmit={(event) => {
            event.preventDefault();
            void sendMessage();
          }}
        >
          <textarea
            placeholder="질문을 입력하세요. Enter로 전송, Shift+Enter로 줄바꿈"
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            onKeyDown={(event) => {
              const isComposing = (event.nativeEvent as KeyboardEvent).isComposing;
              if (isComposing || event.key !== "Enter" || event.shiftKey) {
                return;
              }
              event.preventDefault();
              void sendMessage();
            }}
          />
          <button disabled={isSending || message.trim().length === 0} type="submit">
            보내기
          </button>
        </form>
      </Card>

      <Card title="이용 안내">
        <p className="warning-text">
          AI 건강 상담은 참고용 안내이며, 진단/처방 또는 치료 변경을 대신하지 않습니다. 증상이 있거나 약 복용을
          변경하려면 의료진과 상담해주세요.
        </p>
      </Card>
    </div>
  );
}
