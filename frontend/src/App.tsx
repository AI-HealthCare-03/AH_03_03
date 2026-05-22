import { Route, Routes } from "react-router-dom";

import AdminRoute from "./auth/AdminRoute";
import ProtectedRoute from "./auth/ProtectedRoute";
import AdminLayout from "./components/AdminLayout";
import Layout from "./components/Layout";
import AnalysisHistoryPage from "./pages/AnalysisHistoryPage";
import AnalysisPage from "./pages/AnalysisPage";
import ChallengeDetailPage from "./pages/ChallengeDetailPage";
import ChallengePage from "./pages/ChallengePage";
import ChatbotPage from "./pages/ChatbotPage";
import DashboardPage from "./pages/DashboardPage";
import DietHistoryPage from "./pages/DietHistoryPage";
import DietPage from "./pages/DietPage";
import DietResultPage from "./pages/DietResultPage";
import ExamOcrPage from "./pages/ExamOcrPage";
import FAQPage from "./pages/FAQPage";
import FamilyPage from "./pages/FamilyPage";
import FindLoginIdPage from "./pages/FindLoginIdPage";
import HealthProfilePage from "./pages/HealthProfilePage";
import HealthRecordPage from "./pages/HealthRecordPage";
import InquiryPage from "./pages/InquiryPage";
import LoginPage from "./pages/LoginPage";
import MainPage from "./pages/MainPage";
import MedicationPage from "./pages/MedicationPage";
import MedicationOcrPage from "./pages/MedicationOcrPage";
import MyPage from "./pages/MyPage";
import NotificationPage from "./pages/NotificationPage";
import OcrPage from "./pages/OcrPage";
import PasswordResetConfirmPage from "./pages/PasswordResetConfirmPage";
import PasswordResetRequestPage from "./pages/PasswordResetRequestPage";
import SettingsPage from "./pages/SettingsPage";
import SignupPage from "./pages/SignupPage";
import AdminDashboardPage from "./pages/admin/AdminDashboardPage";
import AdminLogsPage from "./pages/admin/AdminLogsPage";
import AdminMonitoringPage from "./pages/admin/AdminMonitoringPage";

// Wireframe mapping:
// public landing -> MainPage, signup 4-step -> SignupPage, login -> LoginPage,
// logged-in dashboard -> MainPage/DashboardPage, health input -> HealthRecordPage,
// health profile management -> HealthProfilePage,
// analysis result/list -> AnalysisPage/AnalysisHistoryPage, challenge list/detail -> ChallengePage/ChallengeDetailPage,
// diet upload/result/history -> DietPage/DietResultPage/DietHistoryPage, inquiry -> InquiryPage,
// account/settings/notifications/FAQ -> MyPage/SettingsPage/NotificationPage/FAQPage.
export default function App() {
  return (
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
          <Route path="/mypage" element={<MyPage />} />
          <Route path="/health" element={<HealthRecordPage />} />
          <Route path="/health/profile" element={<HealthProfilePage />} />
          <Route path="/ocr" element={<OcrPage />} />
          <Route path="/ocr/exam" element={<ExamOcrPage />} />
          <Route path="/ocr/medication" element={<MedicationOcrPage />} />
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
      </Route>
      <Route element={<AdminRoute />}>
        <Route element={<AdminLayout />}>
          <Route path="/admin" element={<AdminDashboardPage />} />
          <Route path="/admin/monitoring" element={<AdminMonitoringPage />} />
          <Route path="/admin/logs" element={<AdminLogsPage />} />
        </Route>
      </Route>
    </Routes>
  );
}
