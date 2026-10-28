import type { Metadata } from "next";
import { NewsView } from "@/components/views";

export const metadata: Metadata = { title: "News & commentary · Top Gainer Insights" };

export default function NewsPage() {
  return <NewsView />;
}
