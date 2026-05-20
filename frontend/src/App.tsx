import { Route, Routes } from "react-router-dom";

import ProtectedRoute from "./auth/ProtectedRoute";
import Layout from "./components/Layout";
import AnalysisPage from "./pages/AnalysisPage";
import ChallengePage from "./pages/ChallengePage";
import DashboardPage from "./pages/DashboardPage";
import DietPage from "./pages/DietPage";
import FAQPage from "./pages/FAQPage";
import HealthRecordPage from "./pages/HealthRecordPage";
import LoginPage from "./pages/LoginPage";
import MainPage from "./pages/MainPage";
import MedicationPage from "./pages/MedicationPage";
import MyPage from "./pages/MyPage";
import NotificationPage from "./pages/NotificationPage";
import SettingsPage from "./pages/SettingsPage";
import SignupPage from "./pages/SignupPage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<MainPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/signup" element={<SignupPage />} />
        <Route path="/faqs" element={<FAQPage />} />
        <Route element={<ProtectedRoute />}>
          <Route path="/mypage" element={<MyPage />} />
          <Route path="/health" element={<HealthRecordPage />} />
          <Route path="/analysis" element={<AnalysisPage />} />
          <Route path="/challenges" element={<ChallengePage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/diets" element={<DietPage />} />
          <Route path="/medications" element={<MedicationPage />} />
          <Route path="/notifications" element={<NotificationPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Route>
      </Route>
    </Routes>
  );
}
