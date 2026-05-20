import { Route, Routes } from "react-router-dom";

import ProtectedRoute from "./auth/ProtectedRoute";
import Layout from "./components/Layout";
import AnalysisHistoryPage from "./pages/AnalysisHistoryPage";
import AnalysisPage from "./pages/AnalysisPage";
import ChallengeDetailPage from "./pages/ChallengeDetailPage";
import ChallengePage from "./pages/ChallengePage";
import DashboardPage from "./pages/DashboardPage";
import DietHistoryPage from "./pages/DietHistoryPage";
import DietPage from "./pages/DietPage";
import DietResultPage from "./pages/DietResultPage";
import FAQPage from "./pages/FAQPage";
import HealthProfilePage from "./pages/HealthProfilePage";
import HealthRecordPage from "./pages/HealthRecordPage";
import InquiryPage from "./pages/InquiryPage";
import LoginPage from "./pages/LoginPage";
import MainPage from "./pages/MainPage";
import MedicationPage from "./pages/MedicationPage";
import MyPage from "./pages/MyPage";
import NotificationPage from "./pages/NotificationPage";
import SettingsPage from "./pages/SettingsPage";
import SignupPage from "./pages/SignupPage";

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
        <Route path="/faqs" element={<FAQPage />} />
        <Route path="/faq" element={<FAQPage />} />
        <Route element={<ProtectedRoute />}>
          <Route path="/mypage" element={<MyPage />} />
          <Route path="/health" element={<HealthRecordPage />} />
          <Route path="/health/profile" element={<HealthProfilePage />} />
          <Route path="/analysis" element={<AnalysisPage />} />
          <Route path="/analysis/history" element={<AnalysisHistoryPage />} />
          <Route path="/analysis/:analysisId" element={<AnalysisHistoryPage />} />
          <Route path="/challenges" element={<ChallengePage />} />
          <Route path="/challenges/:challengeId" element={<ChallengeDetailPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/diets" element={<DietPage />} />
          <Route path="/diets/history" element={<DietHistoryPage />} />
          <Route path="/diets/:dietRecordId" element={<DietResultPage />} />
          <Route path="/inquiries" element={<InquiryPage />} />
          <Route path="/inquiries/new" element={<InquiryPage />} />
          <Route path="/medications" element={<MedicationPage />} />
          <Route path="/notifications" element={<NotificationPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Route>
      </Route>
    </Routes>
  );
}
