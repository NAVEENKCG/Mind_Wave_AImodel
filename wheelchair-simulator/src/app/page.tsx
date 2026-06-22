"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { motion, AnimatePresence, useReducedMotion } from "framer-motion";
import { Brain, Wifi, WifiOff } from "lucide-react";
import Arena from "@/components/Arena";
import StatusPanel from "@/components/StatusPanel";
import BrainPowerPanel from "@/components/BrainPowerPanel";
import { staggerContainer, fadeInUp, EASE_OUT_EXPO } from "@/lib/animations";

interface EEGData {
  command: string;
  confidence: number;
  signal: string;
  fatigue: string;
  power: { theta: number; alpha: number; beta: number };
}

const GRID_COLS = 24;
const GRID_ROWS = 16;
const WS_URL = "ws://localhost:8765";

export default function Home() {
  const shouldReduceMotion = useReducedMotion();

  const [connected, setConnected] = useState(false);
  const [data, setData] = useState<EEGData>({
    command: "CONNECTING...",
    confidence: 0,
    signal: "AWAITING",
    fatigue: "NORMAL",
    power: { theta: 0, alpha: 0, beta: 0 },
  });

  const [position, setPosition] = useState({ x: GRID_COLS / 2, y: GRID_ROWS / 2 });
  const [totalFwd, setTotalFwd] = useState(0);
  const [totalIdle, setTotalIdle] = useState(0);

  const posRef = useRef(position);
  posRef.current = position;

  // Move wheelchair on FORWARD command
  const handleData = useCallback((msg: EEGData) => {
    setData(msg);
    if (msg.command === "FORWARD") {
      setTotalFwd((n) => n + 1);
      setPosition((p) => {
        const ny = p.y - 1 < 1 ? GRID_ROWS - 2 : p.y - 1;
        return { x: p.x, y: ny };
      });
    } else {
      setTotalIdle((n) => n + 1);
    }
  }, []);

  // WebSocket connection
  useEffect(() => {
    let ws: WebSocket | null = null;
    let retryTimer: ReturnType<typeof setTimeout>;

    const connect = () => {
      try {
        ws = new WebSocket(WS_URL);

        ws.onopen = () => setConnected(true);

        ws.onmessage = (event) => {
          try {
            const msg = JSON.parse(event.data) as EEGData;
            handleData(msg);
          } catch {}
        };

        ws.onerror = () => {
          setConnected(false);
        };

        ws.onclose = () => {
          setConnected(false);
          setData((d) => ({ ...d, command: "OFFLINE", signal: "RECONNECTING..." }));
          retryTimer = setTimeout(connect, 3000);
        };
      } catch {
        retryTimer = setTimeout(connect, 3000);
      }
    };

    connect();
    return () => {
      clearTimeout(retryTimer);
      ws?.close();
    };
  }, [handleData]);

  const motionProps = shouldReduceMotion
    ? { initial: false, animate: false }
    : {};

  return (
    <main
      className="relative min-h-screen flex flex-col"
      style={{ background: "var(--bg-base)" }}
    >
      {/* Skip to content */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-[var(--accent)] focus:text-[var(--bg-base)] focus:rounded-lg focus:font-semibold"
      >
        Skip to content
      </a>

      {/* Navbar */}
      <nav
        className="fixed top-4 left-0 right-0 z-40 flex justify-center px-6"
        aria-label="Main navigation"
      >
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={EASE_OUT_EXPO}
          className="flex items-center gap-4 px-5 py-3 rounded-2xl"
          style={{
            background: "rgba(13,21,38,0.80)",
            border: "1px solid var(--border)",
            backdropFilter: "blur(24px)",
            boxShadow: "inset 0 1px 0 rgba(255,255,255,0.08)",
            maxWidth: "640px",
            width: "100%",
          }}
        >
          <div className="flex items-center gap-2.5 flex-1">
            <div
              className="w-7 h-7 rounded-lg flex items-center justify-center"
              style={{ background: "rgba(56,189,248,0.15)", border: "1px solid rgba(56,189,248,0.3)" }}
            >
              <Brain size={14} color="var(--accent)" />
            </div>
            <span
              className="font-black text-base tracking-tight"
              style={{ fontFamily: "Syne, sans-serif", letterSpacing: "-0.03em" }}
            >
              ORBIT <span style={{ color: "var(--accent)" }}>AI</span>
            </span>
          </div>

          <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-widest font-semibold">
            {connected ? (
              <Wifi size={12} color="#4ADE80" />
            ) : (
              <WifiOff size={12} color="#F87171" />
            )}
            <span style={{ color: connected ? "#4ADE80" : "#F87171" }}>
              {connected ? "Connected" : "Offline"}
            </span>
          </div>
        </motion.div>
      </nav>

      {/* Main Content */}
      <div
        id="main-content"
        className="flex-1 flex flex-col pt-24 pb-8 px-6 gap-6 relative z-10"
        style={{ maxWidth: "1200px", margin: "0 auto", width: "100%" }}
      >
        {/* Header */}
        <motion.header
          variants={staggerContainer}
          initial="hidden"
          animate="visible"
          className="flex flex-col gap-1"
          {...motionProps}
        >
          <motion.p
            variants={fadeInUp}
            className="text-[10px] uppercase tracking-[0.2em] font-semibold"
            style={{ color: "var(--accent)", opacity: 0.7 }}
          >
            Brain-Computer Interface
          </motion.p>
          <motion.h1
            variants={fadeInUp}
            className="text-5xl md:text-6xl font-black leading-none"
            style={{
              fontFamily: "Syne, sans-serif",
              letterSpacing: "-0.04em",
              background: "linear-gradient(135deg, #fff 30%, rgba(56,189,248,0.8) 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            Virtual Wheelchair
          </motion.h1>
          <motion.p
            variants={fadeInUp}
            className="text-sm max-w-md"
            style={{ color: "var(--text-secondary)" }}
          >
            EEG-controlled simulation via NeuroSky TGAM · MOABB CSP+LDA AI Model
          </motion.p>
        </motion.header>

        {/* Dashboard Grid */}
        <motion.div
          variants={staggerContainer}
          initial="hidden"
          animate="visible"
          className="grid gap-4"
          style={{
            gridTemplateColumns: "280px 1fr",
            gridTemplateRows: "auto",
          }}
          {...motionProps}
        >
          {/* Left Sidebar */}
          <motion.aside
            variants={fadeInUp}
            className="flex flex-col gap-4"
            style={{ gridColumn: "1", gridRow: "1" }}
          >
            <StatusPanel
              command={data.command}
              confidence={data.confidence}
              signal={data.signal}
              fatigue={data.fatigue}
              totalFwd={totalFwd}
              totalIdle={totalIdle}
              connected={connected}
            />
            <BrainPowerPanel power={data.power} />

            {/* Info Card */}
            <motion.div
              variants={fadeInUp}
              className="rounded-2xl p-4"
              style={{
                background: "var(--glass-bg)",
                border: "1px solid var(--border)",
                boxShadow: "inset 0 1px 0 rgba(255,255,255,0.10)",
              }}
            >
              <p className="text-[10px] uppercase tracking-widest font-semibold mb-2" style={{ color: "var(--accent)", opacity: 0.7 }}>
                How it works
              </p>
              <ol className="flex flex-col gap-1.5">
                {[
                  "Run simulate_tgam.py",
                  "Run predict_websocket.py",
                  "Watch the wheelchair move!",
                ].map((step, i) => (
                  <li key={i} className="flex items-start gap-2 text-[11px]" style={{ color: "var(--text-secondary)" }}>
                    <span
                      className="text-[9px] font-bold rounded-full w-4 h-4 flex items-center justify-center flex-shrink-0 mt-px"
                      style={{ background: "rgba(56,189,248,0.15)", color: "var(--accent)", border: "1px solid rgba(56,189,248,0.2)" }}
                    >
                      {i + 1}
                    </span>
                    {step}
                  </li>
                ))}
              </ol>
            </motion.div>
          </motion.aside>

          {/* Arena */}
          <motion.section
            variants={fadeInUp}
            aria-label="Virtual Wheelchair Arena"
            style={{ gridColumn: "2", gridRow: "1", minHeight: "520px" }}
          >
            <Arena
              command={data.command}
              position={position}
              fatigue={data.fatigue}
            />
          </motion.section>
        </motion.div>

        {/* Offline Banner */}
        <AnimatePresence>
          {!connected && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              className="fixed bottom-6 left-0 right-0 flex justify-center px-6 z-30"
            >
              <div
                className="flex items-center gap-3 px-5 py-3 rounded-2xl text-sm"
                style={{
                  background: "rgba(13,21,38,0.95)",
                  border: "1px solid rgba(248,113,113,0.35)",
                  backdropFilter: "blur(20px)",
                  borderLeft: "3px solid #F87171",
                }}
              >
                <WifiOff size={14} color="#F87171" />
                <span style={{ color: "var(--text-secondary)" }}>
                  Dashboard offline. Start{" "}
                  <code
                    className="px-1.5 py-0.5 rounded text-[11px] font-mono"
                    style={{ background: "rgba(255,255,255,0.08)", color: "#F87171" }}
                  >
                    predict_websocket.py
                  </code>{" "}
                  and retry.
                </span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </main>
  );
}
