import type { Metadata } from "next";
import { ReportsView } from "@/components/views";

export const metadata: Metadata = { title: "Reports · Top Gainer Insights" };

export default function ReportsPage() {
  return <ReportsView />;
}
