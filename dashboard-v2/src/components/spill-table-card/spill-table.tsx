import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';

import { useSpillEvents } from '@/hooks';
import { DataTable } from './data-table';
import { columns } from './columns';

export function SpillTableCard() {
  const { data } = useSpillEvents();

  return (
    <Card>
      <CardHeader>
        <CardTitle>Storm Drain Outflows</CardTitle>
        <CardDescription>
          These events can indicate the spillage of sewage into the river.
        </CardDescription>
      </CardHeader>
      <DataTable columns={columns} data={data || []} loading={!data} />
    </Card>
  );
}
