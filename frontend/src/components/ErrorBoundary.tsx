import { Component, type ErrorInfo, type ReactNode } from "react";

type ErrorBoundaryProps = {
  children: ReactNode;
};

type ErrorBoundaryState = {
  error: Error | null;
};

export default class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = {
    error: null,
  };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    if (import.meta.env.DEV) {
      console.error("Render error boundary caught an error", error, errorInfo);
    }
  }

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (!this.state.error) {
      return this.props.children;
    }

    return (
      <main className="error-page-shell">
        <section className="error-page-card">
          <div className="error-page-mark">!</div>
          <h1>문제가 발생했습니다.</h1>
          <p>화면을 불러오는 중 오류가 발생했습니다. 홈으로 이동하거나 새로고침해 다시 시도해주세요.</p>
          {import.meta.env.DEV ? <pre>{this.state.error.message}</pre> : null}
          <div className="button-row">
            <a className="button" href="/">
              홈으로 이동
            </a>
            <button className="button secondary" onClick={this.handleReload} type="button">
              새로고침
            </button>
          </div>
        </section>
      </main>
    );
  }
}
