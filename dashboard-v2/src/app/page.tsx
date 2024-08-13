import { LevelChartCard } from "@/components/level-chart-card";
import { ParametersCard } from '@/components/parameters-card';
import { SpillTableCard } from "@/components/spill-table-card";

export default function Index() {
  return (
    <main>
      <div className="bg-secondary p-4 mb-4 text-center rounded-sm">
        <span className="font-bold text-lg text-secondary-foreground">
          🚧 This service is still a work in progress. Please do not rely on any
          data from it! 🚧
        </span>
      </div>

      <div className="container px-2">
        <div className="grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-4">
          <div className="col-span-1 md:col-span-3 h-auto row-span-3">
            <LevelChartCard />
          </div>
          <div className="col-span-1 row-span-4">
            <ParametersCard />
          </div>
          <div className="col-span-1 md:col-span-3 row-span-2">
            <SpillTableCard />
          </div>
        </div>
      </div>
    </main>
  );
}
