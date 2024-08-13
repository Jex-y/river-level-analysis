import {
  TableCell,
  TableRow,
} from '@/components/ui/table';

import { Skeleton } from '@/components/ui/skeleton';
import type { ColumnDef, Table as TableType } from '@tanstack/react-table';
import type { SpillEvent } from '@/types';
import { flexRender } from '@tanstack/react-table';

type SkeletonRowsProps = {
  columns: ColumnDef<SpillEvent>[];
  count: number;
};
function SkeletonRows({ columns, count }: SkeletonRowsProps) {
  return [...Array(count)].map((_, index) => (
    // biome-ignore lint/suspicious/noArrayIndexKey: Shouldn't need to update these at all
    <TableRow key={index}>
      {columns.map((column) => (
        <TableCell key={column.id}>
          <Skeleton className="h-2 w-full" />
        </TableCell>
      ))}
    </TableRow>
  ));
}


type EmptyTableProps = {
  columns: ColumnDef<SpillEvent>[];
  empty_message?: string;
};

function EmptyTable({
  columns,
  empty_message = 'No results.',
}: EmptyTableProps) {
  return (
    <TableRow>
      <TableCell
        colSpan={columns.length}
        className="h-24 text-center"
      >
        {empty_message}
      </TableCell>
    </TableRow>
  );
}

type TableRowsProps = {
  table: TableType<SpillEvent>;
};

function TableRows({ table }: TableRowsProps) {
  return table.getRowModel().rows.map((row) => (
    <TableRow
      key={row.id}
      data-state={row.getIsSelected() && 'selected'}
    >
      {row.getVisibleCells().map((cell) => (
        <TableCell key={cell.id}>
          {flexRender(cell.column.columnDef.cell, cell.getContext())}
        </TableCell>
      ))}
    </TableRow>
  ))
}

type TableContentProps = {
  table: TableType<SpillEvent>;
  columns: ColumnDef<SpillEvent>[];
  loading: boolean;
  pageSize: number;
};

export function TableContent({ table, columns, pageSize, loading }: TableContentProps) {
  if (table.getRowCount() === 0) {
    return <EmptyTable columns={columns} />;
  }

  if (loading) {
    return <SkeletonRows columns={columns} count={pageSize} />;
  }

  return <TableRows table={table} />;
}
