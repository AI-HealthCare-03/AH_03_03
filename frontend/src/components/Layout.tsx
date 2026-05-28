import { useCallback, useState } from "react";
import { Outlet, useLocation } from "react-router-dom";

import MobileNav from "./MobileNav";
import Navbar from "./Navbar";
import Sidebar from "./Sidebar";
import { useAuth } from "../auth/AuthContext";

export default function Layout() {
  const { isAuthenticated } = useAuth();
  const location = useLocation();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const isAuthPage = location.pathname === "/login" || location.pathname === "/signup";
  const showSidebar = isAuthenticated && !isAuthPage;
  const closeMobileMenu = useCallback(() => setIsMobileMenuOpen(false), []);

  return (
    <div className={showSidebar ? "app-shell" : "app-shell app-shell-public"}>
      <Navbar
        isMobileMenuOpen={isMobileMenuOpen}
        onMobileMenuOpen={() => setIsMobileMenuOpen(true)}
        showMobileMenuButton={showSidebar}
      />
      {showSidebar && <Sidebar />}
      {showSidebar && <MobileNav isOpen={isMobileMenuOpen} onClose={closeMobileMenu} />}
      <main className={showSidebar ? "page" : "page page-public"}>
        <Outlet />
      </main>
    </div>
  );
}
