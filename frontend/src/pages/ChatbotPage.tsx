import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";

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

function isChatbotContextType(value: string | null): value is ChatbotContextType {
  return Boolean(value && ["MAIN", "ANALYSIS", "DIET", "CHALLENGE", "GENERAL"].includes(value));
}

function toUserFacingText(text: string) {
  return text;
}

const chatSafetyNotice = "AI 상담은 건강관리 참고용이며, 진단·치료는 의료진 상담이 필요합니다.";

export default function ChatbotPage() {
  const location = useLocation();
  const [selectedContext, setSelectedContext] = useState<ContextOption>(contextOptions[0]);
  const [targetId, setTargetId] = useState<number | null>(null);
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
        target_id: targetId,
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

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const state = (location.state ?? {}) as {
      context_type?: ChatbotContextType;
      target_id?: number | string | null;
      initial_question?: string;
    };
    const contextValue = isChatbotContextType(params.get("context_type"))
      ? params.get("context_type")
      : state.context_type;
    if (isChatbotContextType(contextValue ?? null)) {
      const matched = contextOptions.find((option) => option.value === contextValue);
      if (matched) {
        setSelectedContext(matched);
      }
    }
    const rawTargetId = params.get("target_id") ?? state.target_id;
    const parsedTargetId = rawTargetId === null || rawTargetId === undefined ? NaN : Number(rawTargetId);
    setTargetId(Number.isFinite(parsedTargetId) && parsedTargetId > 0 ? parsedTargetId : null);
    const initialQuestion = params.get("initial_question") ?? state.initial_question;
    if (initialQuestion) {
      setMessage(initialQuestion);
    }
  }, [location.search, location.state]);

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
        <div className="state-box" style={{ marginBottom: "12px" }}>
          {chatSafetyNotice}
        </div>
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

      <Card title="AI 건강 대화">
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
              {chatMessage.role === "assistant" && (
                <div className="chat-avatar">
                  <span>H</span>
                </div>
              )}
              <div className="chat-bubble-wrapper">
                <span className="chat-role">{chatMessage.role === "user" ? "나" : "AI 건강 상담"}</span>
                <div className="chat-bubble">
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
                </div>
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
        <p className="muted" style={{ fontSize: "13px", marginTop: 8, textAlign: "left" }}>
          *서비스 이용 관련 문의는 <Link to="/inquiries" style={{ color: "inherit", textDecoration: "underline" }}>문의/FAQ</Link>를 이용해 주세요.
        </p>
      </Card>
    </div>
  );
}
