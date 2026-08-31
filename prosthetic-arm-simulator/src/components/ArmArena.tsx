"use client";

import { motion, useReducedMotion } from "framer-motion";

interface ArmArenaProps {
  command: string;
  fatigue: string;
}

/**
 * ArmArena — Animated SVG robotic prosthetic arm visualization.
 *
 * Commands:
 * - IDLE:  Arm at rest, neutral glow, fingers relaxed
 * - LIFT:  Arm raises upward with green pulse glow
 */
export function ArmArena({ command, fatigue }: ArmArenaProps) {
  const shouldReduceMotion = useReducedMotion();

  const isLift = command === "LIFT";
  const isCritical = fatigue === "CRITICAL";

  const borderColor = isCritical
    ? "rgba(248, 113, 113, 0.5)"
    : isLift
    ? "rgba(74, 222, 128, 0.4)"
    : "rgba(255,255,255,0.08)";

  const glowColor = isCritical
    ? "rgba(248, 113, 113, 0.1)"
    : isLift
    ? "rgba(74, 222, 128, 0.08)"
    : "transparent";

  // Arm rotation: 0° = resting (pointing down-right), -35° = lifted
  const armRotation = isLift ? -35 : 0;

  // Finger splay: fingers spread slightly when lifting
  const fingerSplay = isLift ? 8 : 0;

  // Accent color for the arm based on state
  const armColor = isCritical
    ? "#F87171"
    : isLift
    ? "#4ADE80"
    : "#2DD4BF";

  const jointColor = isCritical
    ? "#EF4444"
    : isLift
    ? "#22C55E"
    : "#14B8A6";

  return (
    <div
      className="relative w-full h-full rounded-3xl overflow-hidden flex items-center justify-center"
      style={{
        background: `radial-gradient(ellipse at center, ${glowColor} 0%, transparent 70%), var(--bg-surface)`,
        border: `1px solid ${borderColor}`,
        boxShadow: `inset 0 1px 0 rgba(255,255,255,0.10), 0 0 40px ${glowColor}`,
        transition: "border-color 0.4s ease, box-shadow 0.4s ease",
      }}
    >
      {/* Grid background */}
      <svg
        className="absolute inset-0 w-full h-full opacity-[0.04]"
        style={{ pointerEvents: "none" }}
      >
        <defs>
          <pattern
            id="arm-grid"
            width="4.1667%"
            height="6.25%"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M 40 0 L 0 0 0 40"
              fill="none"
              stroke="white"
              strokeWidth="1"
            />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#arm-grid)" />
      </svg>

      {/* Corner decorations */}
      {["top-2 left-2", "top-2 right-2", "bottom-2 left-2", "bottom-2 right-2"].map((pos, i) => (
        <div
          key={i}
          className={`absolute ${pos} w-4 h-4 opacity-30`}
          style={{
            borderTop: i < 2 ? "1px solid var(--accent)" : "none",
            borderBottom: i >= 2 ? "1px solid var(--accent)" : "none",
            borderLeft: i % 2 === 0 ? "1px solid var(--accent)" : "none",
            borderRight: i % 2 === 1 ? "1px solid var(--accent)" : "none",
          }}
        />
      ))}

      {/* Pulse ring behind arm when lifting */}
      <motion.div
        className="absolute rounded-full"
        animate={{
          scale: isLift ? [1, 2.0, 1] : 1,
          opacity: isLift ? [0.4, 0, 0.4] : 0,
        }}
        transition={shouldReduceMotion
          ? { duration: 0 }
          : { duration: 1.8, repeat: Infinity, ease: "easeOut" }
        }
        style={{
          width: 180,
          height: 180,
          background: `radial-gradient(circle, rgba(74,222,128,0.2), transparent)`,
        }}
      />

      {/* Secondary pulse ring — offset timing */}
      <motion.div
        className="absolute rounded-full"
        animate={{
          scale: isLift ? [1, 1.6, 1] : 1,
          opacity: isLift ? [0.25, 0, 0.25] : 0,
        }}
        transition={shouldReduceMotion
          ? { duration: 0 }
          : { duration: 1.8, repeat: Infinity, ease: "easeOut", delay: 0.6 }
        }
        style={{
          width: 120,
          height: 120,
          background: `radial-gradient(circle, rgba(45,212,191,0.2), transparent)`,
        }}
      />

      {/* Robotic Arm SVG */}
      <motion.svg
        width="280"
        height="360"
        viewBox="0 0 280 360"
        fill="none"
        animate={{
          filter: isLift
            ? "drop-shadow(0 0 16px rgba(74,222,128,0.6))"
            : isCritical
            ? "drop-shadow(0 0 12px rgba(248,113,113,0.6))"
            : "drop-shadow(0 0 6px rgba(45,212,191,0.3))",
        }}
        transition={{ duration: 0.4 }}
        style={{ position: "relative", zIndex: 10 }}
      >
        {/* === UPPER ARM (shoulder to elbow) === */}
        <motion.g
          animate={{ rotate: shouldReduceMotion ? 0 : armRotation }}
          transition={{ type: "spring", stiffness: 200, damping: 25 }}
          style={{ originX: "140px", originY: "40px" }}
        >
          {/* Shoulder joint */}
          <motion.circle
            cx="140"
            cy="40"
            r="16"
            stroke={jointColor}
            strokeWidth="2.5"
            fill="none"
            animate={{ r: isLift ? 18 : 16 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
          />
          <circle cx="140" cy="40" r="6" fill={jointColor} opacity={0.6} />

          {/* Upper arm segment */}
          <motion.rect
            x="132"
            y="56"
            width="16"
            rx="6"
            stroke={armColor}
            strokeWidth="2"
            fill="rgba(255,255,255,0.03)"
            animate={{ height: 80 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
          />

          {/* Upper arm panel lines */}
          <line x1="136" y1="70" x2="144" y2="70" stroke={armColor} strokeWidth="1" opacity={0.3} />
          <line x1="136" y1="85" x2="144" y2="85" stroke={armColor} strokeWidth="1" opacity={0.3} />
          <line x1="136" y1="100" x2="144" y2="100" stroke={armColor} strokeWidth="1" opacity={0.3} />
          <line x1="136" y1="115" x2="144" y2="115" stroke={armColor} strokeWidth="1" opacity={0.3} />

          {/* Elbow joint */}
          <motion.circle
            cx="140"
            cy="146"
            r="12"
            stroke={jointColor}
            strokeWidth="2"
            fill="none"
            animate={{ r: isLift ? 14 : 12 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
          />
          <circle cx="140" cy="146" r="4" fill={jointColor} opacity={0.5} />

          {/* === FOREARM (elbow to wrist) === */}
          <rect
            x="134"
            y="158"
            width="12"
            height="70"
            rx="5"
            stroke={armColor}
            strokeWidth="2"
            fill="rgba(255,255,255,0.03)"
          />

          {/* Forearm panel lines */}
          <line x1="138" y1="172" x2="142" y2="172" stroke={armColor} strokeWidth="1" opacity={0.3} />
          <line x1="138" y1="187" x2="142" y2="187" stroke={armColor} strokeWidth="1" opacity={0.3} />
          <line x1="138" y1="202" x2="142" y2="202" stroke={armColor} strokeWidth="1" opacity={0.3} />

          {/* Wrist joint */}
          <motion.circle
            cx="140"
            cy="238"
            r="10"
            stroke={jointColor}
            strokeWidth="2"
            fill="none"
            animate={{ r: isLift ? 12 : 10 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
          />
          <circle cx="140" cy="238" r="3.5" fill={jointColor} opacity={0.5} />

          {/* === PALM === */}
          <rect
            x="122"
            y="250"
            width="36"
            height="28"
            rx="8"
            stroke={armColor}
            strokeWidth="2"
            fill="rgba(255,255,255,0.04)"
          />

          {/* Palm circuit lines */}
          <line x1="132" y1="258" x2="148" y2="258" stroke={armColor} strokeWidth="0.8" opacity={0.25} />
          <line x1="132" y1="266" x2="148" y2="266" stroke={armColor} strokeWidth="0.8" opacity={0.25} />

          {/* === FINGERS === */}
          {/* Thumb (angled left) */}
          <motion.g
            animate={{ rotate: shouldReduceMotion ? 0 : isLift ? -15 : 0 }}
            style={{ originX: "122px", originY: "260px" }}
          >
            <rect x="102" y="258" width="20" height="8" rx="4" stroke={armColor} strokeWidth="1.5" fill="rgba(255,255,255,0.03)" />
            <circle cx="102" cy="262" r="3" fill={jointColor} opacity={0.4} />
          </motion.g>

          {/* Index finger */}
          <motion.g
            animate={{ rotate: shouldReduceMotion ? 0 : -fingerSplay }}
            style={{ originX: "128px", originY: "278px" }}
          >
            <rect x="124" y="278" width="8" height="36" rx="4" stroke={armColor} strokeWidth="1.5" fill="rgba(255,255,255,0.03)" />
            <circle cx="128" cy="314" r="2.5" fill={jointColor} opacity={0.4} />
          </motion.g>

          {/* Middle finger */}
          <motion.g
            animate={{ rotate: shouldReduceMotion ? 0 : -(fingerSplay * 0.3) }}
            style={{ originX: "140px", originY: "278px" }}
          >
            <rect x="136" y="278" width="8" height="40" rx="4" stroke={armColor} strokeWidth="1.5" fill="rgba(255,255,255,0.03)" />
            <circle cx="140" cy="318" r="2.5" fill={jointColor} opacity={0.4} />
          </motion.g>

          {/* Ring finger */}
          <motion.g
            animate={{ rotate: shouldReduceMotion ? 0 : fingerSplay * 0.3 }}
            style={{ originX: "152px", originY: "278px" }}
          >
            <rect x="148" y="278" width="8" height="36" rx="4" stroke={armColor} strokeWidth="1.5" fill="rgba(255,255,255,0.03)" />
            <circle cx="152" cy="314" r="2.5" fill={jointColor} opacity={0.4} />
          </motion.g>

          {/* Pinky finger */}
          <motion.g
            animate={{ rotate: shouldReduceMotion ? 0 : fingerSplay }}
            style={{ originX: "158px", originY: "278px" }}
          >
            <rect x="156" y="278" width="7" height="30" rx="3.5" stroke={armColor} strokeWidth="1.5" fill="rgba(255,255,255,0.03)" />
            <circle cx="159.5" cy="308" r="2" fill={jointColor} opacity={0.4} />
          </motion.g>
        </motion.g>

        {/* Shoulder mount — stays fixed */}
        <rect x="120" y="20" width="40" height="8" rx="4" fill={armColor} opacity={0.3} />
        <line x1="115" y1="24" x2="165" y2="24" stroke={armColor} strokeWidth="1" opacity={0.2} />
      </motion.svg>

      {/* Status overlay */}
      <div className="absolute bottom-3 left-0 right-0 flex justify-center">
        <motion.div
          key={command}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          className="px-4 py-1 rounded-full text-xs font-semibold tracking-widest uppercase"
          style={{
            background: isCritical
              ? "rgba(248,113,113,0.15)"
              : isLift
              ? "rgba(74,222,128,0.12)"
              : "rgba(255,255,255,0.05)",
            border: `1px solid ${isCritical ? "rgba(248,113,113,0.4)" : isLift ? "rgba(74,222,128,0.35)" : "rgba(255,255,255,0.10)"}`,
            color: isCritical ? "#F87171" : isLift ? "#4ADE80" : "rgba(255,255,255,0.5)",
          }}
        >
          {command}
        </motion.div>
      </div>

      {/* State indicator label — top left */}
      <div className="absolute top-4 left-4 flex items-center gap-2">
        <motion.div
          className="w-2 h-2 rounded-full"
          animate={{
            background: isLift ? "#4ADE80" : isCritical ? "#F87171" : "rgba(45,212,191,0.6)",
            scale: isLift ? [1, 1.3, 1] : 1,
          }}
          transition={shouldReduceMotion
            ? { duration: 0 }
            : { duration: 1, repeat: isLift ? Infinity : 0 }
          }
        />
        <span
          className="text-[10px] uppercase tracking-widest font-semibold"
          style={{ color: isLift ? "#4ADE80" : isCritical ? "#F87171" : "var(--text-muted)" }}
        >
          {isLift ? "ARM LIFTING" : isCritical ? "FATIGUE DETECTED" : "ARM AT REST"}
        </span>
      </div>
    </div>
  );
}

export default ArmArena;
