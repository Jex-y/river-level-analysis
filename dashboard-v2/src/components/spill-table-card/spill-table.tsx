import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';

import { useSpillSites } from '@/hooks';
import { columns } from './columns';
import { DataTable } from './data-table';

export function SpillTableCard() {
  const { data } = useSpillSites();

  return (
    <Card>
      <CardHeader>
        <CardTitle>Storm Drain Outflows Sites</CardTitle>
        <CardDescription>
          Storm drain outflow sites along the River Wear.
        </CardDescription>
      </CardHeader>
      <DataTable columns={columns} data={data || []} loading={!data} />
    </Card>
  );
}
