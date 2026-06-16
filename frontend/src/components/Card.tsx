import type { ReactNode } from "react";

type CardProps = {
  title?: string;
  children: ReactNode;
  actions?: ReactNode;
  className?: string;
};

export default function Card({ title, children, actions, className }: CardProps) {
  return (
    <section className={`card${className ? ` ${className}` : ""}`}>
      {(title || actions) && (
        <div className="card-header">
          {title && <h2>{title}</h2>}
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}
