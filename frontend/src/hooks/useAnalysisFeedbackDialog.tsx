import { useCallback, useMemo, useState } from "react";

import ConfirmDialog from "../components/ConfirmDialog";

type AnalysisFeedbackStatus = "processing" | "success" | "error" | "info";

type AnalysisFeedbackOptions = {
  message?: string;
  title?: string;
};

type AnalysisFeedbackState = {
  message: string;
  status: AnalysisFeedbackStatus;
  title: string;
};

const DEFAULT_PROCESSING_MESSAGE = "잠시만 기다려 주세요.";
const DEFAULT_SUCCESS_MESSAGE = "결과를 확인해 주세요.";
const DEFAULT_FAILURE_MESSAGE = "다시 시도해주세요.";

export function useAnalysisFeedbackDialog() {
  const [feedback, setFeedback] = useState<AnalysisFeedbackState | null>(null);

  const clearFeedback = useCallback(() => {
    setFeedback(null);
  }, []);

  const showFeedback = useCallback((status: AnalysisFeedbackStatus, options: AnalysisFeedbackOptions) => {
    setFeedback({
      message: options.message ?? "",
      status,
      title: options.title ?? "",
    });
  }, []);

  const showProcessing = useCallback((options: AnalysisFeedbackOptions = {}) => {
    showFeedback("processing", {
      message: options.message ?? DEFAULT_PROCESSING_MESSAGE,
      title: options.title ?? "분석 중입니다...",
    });
  }, [showFeedback]);

  const showSuccess = useCallback((options: AnalysisFeedbackOptions = {}) => {
    showFeedback("success", {
      message: options.message ?? DEFAULT_SUCCESS_MESSAGE,
      title: options.title ?? "분석 완료되었습니다.",
    });
  }, [showFeedback]);

  const showFailure = useCallback((options: AnalysisFeedbackOptions = {}) => {
    showFeedback("error", {
      message: options.message ?? DEFAULT_FAILURE_MESSAGE,
      title: options.title ?? "분석에 실패했습니다.",
    });
  }, [showFeedback]);

  const feedbackDialog = useMemo(() => {
    if (!feedback) {
      return null;
    }
    return (
      <ConfirmDialog
        confirmLabel="확인"
        message={feedback.message}
        onConfirm={clearFeedback}
        showActions={feedback.status !== "processing"}
        showCancel={false}
        title={feedback.title}
        tone={feedback.status === "error" ? "danger" : "default"}
      />
    );
  }, [clearFeedback, feedback]);

  return {
    clearFeedback,
    feedback,
    feedbackDialog,
    showFailure,
    showFeedback,
    showProcessing,
    showSuccess,
  };
}
