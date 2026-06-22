import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ORBIT AI — Virtual EEG Wheelchair Simulator",
  description:
    "Brain-Computer Interface dashboard for EEG-controlled virtual wheelchair simulation using NeuroSky TGAM, MOABB CSP+LDA AI model.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
