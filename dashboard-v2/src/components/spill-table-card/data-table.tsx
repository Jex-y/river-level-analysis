import {
  CardContent,
  CardFooter,
} from '@/components/ui/card';

import {
  type ColumnDef,
  flexRender,
} from '@tanstack/react-table';

import {
  Table,
  TableBody,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';

import type { SpillEvent } from '@/types';
import { TableContent } from './table-content';
import { useDataTable } from './useDataTable';
import { PaginationControls, PaginationInfo } from './pagination';


interface DataTableProps {
  columns: ColumnDef<SpillEvent>[];
  data: SpillEvent[];
  empty_message?: string;
  loading?: boolean;
}

export function DataTable({
  columns,
  data,
  loading = false,
}: DataTableProps) {

  const { table, pagination } = useDataTable(data, columns);

  return (
    <>
      <CardContent>
        <div className="rounded-md border border-border">
          <Table>
            <TableHeader>
              {table.getHeaderGroups().map((headerGroup) => (
                <TableRow key={headerGroup.id}>
                  {headerGroup.headers.map((header) => {
                    return (
                      <TableHead key={header.id}>
                        {header.isPlaceholder
                          ? null
                          : flexRender(
                            header.column.columnDef.header,
                            header.getContext()
                          )}
                      </TableHead>
                    );
                  })}
                </TableRow>
              ))}
            </TableHeader>
            <TableBody>
              <TableContent table={table} columns={columns} loading={loading} pageSize={pagination.pageSize} />
            </TableBody>
          </Table>
        </div>
      </CardContent>
      <CardFooter>
        <div className="flex items-center justify-between w-full">
          <PaginationInfo table={table} pagination={pagination} />
          <PaginationControls table={table} />
        </div>
      </CardFooter>
    </>
  );
}