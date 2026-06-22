export const SPRING_SMOOTH = { type: "spring", stiffness: 300, damping: 30 };
export const SPRING_SNAPPY = { type: "spring", stiffness: 500, damping: 35 };
export const EASE_OUT_EXPO = { duration: 0.6, ease: [0.16, 1, 0.3, 1] as const };

export const fadeInUp = {
  hidden: { opacity: 0, y: 40 },
  visible: { opacity: 1, y: 0, transition: EASE_OUT_EXPO },
};

export const staggerContainer = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.08 } },
};
