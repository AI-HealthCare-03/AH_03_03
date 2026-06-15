import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";

import ProtectedRoute from "./auth/ProtectedRoute";
import ErrorBoundary from "./components/ErrorBoundary";
import Layout from "./components/Layout";
import ScrollToTop from "./components/ScrollToTop";
import Loading from "./components/Loading";

const AboutPage = lazy(() => import("./pages/AboutPage"));
const AnalysisHistoryPage = lazy(() => import("./pages/AnalysisHistoryPage"));
const AnalysisPage = lazy(() => import("./pages/AnalysisPage"));
const ChallengeDetailPage = lazy(() => import("./pages/ChallengeDetailPage"));
const ChallengePage = lazy(() => import("./pages/ChallengePage"));
const ChatbotPage = lazy(() => import("./pages/ChatbotPage"));
const DashboardPage = lazy(() => import("./pages/DashboardPage"));
const DietHistoryPage = lazy(() => import("./pages/DietHistoryPage"));
const DietPage = lazy(() => import("./pages/DietPage"));
const DietResultPage = lazy(() => import("./pages/DietResultPage"));
const ExamOcrPage = lazy(() => import("./pages/ExamOcrPage"));
const FAQPage = lazy(() => import("./pages/FAQPage"));
const FamilyPage = lazy(() => import("./pages/FamilyPage"));
const FindLoginIdPage = lazy(() => import("./pages/FindLoginIdPage"));
const HealthProfilePage = lazy(() => import("./pages/HealthProfilePage"));
const HealthRecordPage = lazy(() => import("./pages/HealthRecordPage"));
const InquiryPage = lazy(() => import("./pages/InquiryPage"));
const LoginPage = lazy(() => import("./pages/LoginPage"));
const MainPage = lazy(() => import("./pages/MainPage"));
const MedicationPage = lazy(() => import("./pages/MedicationPage"));
const MyPage = lazy(() => import("./pages/MyPage"));
const NotFoundPage = lazy(() => import("./pages/NotFoundPage"));
const NotificationPage = lazy(() => import("./pages/NotificationPage"));
const PasswordResetConfirmPage = lazy(() => import("./pages/PasswordResetConfirmPage"));
const PasswordResetRequestPage = lazy(() => import("./pages/PasswordResetRequestPage"));
const SettingsPage = lazy(() => import("./pages/SettingsPage"));
const SignupPage = lazy(() => import("./pages/SignupPage"));

// Wireframe mapping:
// public landing -> MainPage, signup 4-step -> SignupPage, login -> LoginPage,
// logged-in dashboard -> MainPage/DashboardPage, health input -> HealthRecordPage,
// health profile management -> HealthProfilePage,
// analysis result/list -> AnalysisPage/AnalysisHistoryPage, challenge list/detail -> ChallengePage/ChallengeDetailPage,
// diet upload/result/history -> DietPage/DietResultPage/DietHistoryPage, inquiry -> InquiryPage,
// account/settings/notifications/FAQ -> MyPage/SettingsPage/NotificationPage/FAQPage.
export default function App() {
  return (
    <ErrorBoundary>
      <Suspense fallback={<Loading />}>
        <ScrollToTop />
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<MainPage />} />
            <Route path="/login" element={<LoginPage />} />
            <Route path="/signup" element={<SignupPage />} />
            <Route path="/auth/find-login-id" element={<FindLoginIdPage />} />
            <Route path="/auth/password-reset" element={<PasswordResetRequestPage />} />
            <Route path="/auth/password-reset/confirm" element={<PasswordResetConfirmPage />} />
            <Route path="/faqs" element={<FAQPage />} />
            <Route path="/faq" element={<FAQPage />} />
            <Route element={<ProtectedRoute />}>
              <Route path="/about" element={<AboutPage />} />
              <Route path="/mypage" element={<MyPage />} />
              <Route path="/health" element={<HealthRecordPage />} />
              <Route path="/health/profile" element={<HealthProfilePage />} />
              <Route path="/ocr/exam" element={<ExamOcrPage />} />
              <Route path="/chatbot" element={<ChatbotPage />} />
              <Route path="/analysis" element={<AnalysisPage />} />
              <Route path="/analysis/history" element={<AnalysisHistoryPage />} />
              <Route path="/analysis/:analysisId" element={<AnalysisHistoryPage />} />
              <Route path="/challenges" element={<ChallengePage />} />
              <Route path="/challenges/:challengeId" element={<ChallengeDetailPage />} />
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/diets" element={<DietPage />} />
              <Route path="/diets/history" element={<DietHistoryPage />} />
              <Route path="/diets/:dietRecordId" element={<DietResultPage />} />
              <Route path="/family" element={<FamilyPage />} />
              <Route path="/inquiries" element={<InquiryPage />} />
              <Route path="/inquiries/new" element={<InquiryPage />} />
              <Route path="/medications" element={<MedicationPage />} />
              <Route path="/notifications" element={<NotificationPage />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Route>
            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </Suspense>
    </ErrorBoundary>
  );
}
