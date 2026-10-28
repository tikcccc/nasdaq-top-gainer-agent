import type { Metadata } from "next";
import { AnalysisView } from "@/components/views";

export const metadata: Metadata = { title: "Analysis · Top Gainer Insights" };

export default function AnalysisPage() {
  return <AnalysisView />;
}
