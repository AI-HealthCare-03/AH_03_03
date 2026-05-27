import { Outlet, useLocation } from "react-router-dom";

import Navbar from "./Navbar";
import Sidebar from "./Sidebar";
import { useAuth } from "../auth/AuthContext";

export default function Layout() {
  const { isAuthenticated } = useAuth();
  const location = useLocation();
  const isAuthPage = location.pathname === "/login" || location.pathname === "/signup";
  const showSidebar = isAuthenticated && !isAuthPage;

  return (
    <div className={showSidebar ? "app-shell" : "app-shell app-shell-public"}>
      <Navbar />
      {showSidebar && <Sidebar />}
      <main className={showSidebar ? "page" : "page page-public"}>
        <Outlet />
      </main>
    </div>
  );
}
