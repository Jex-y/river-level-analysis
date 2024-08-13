import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { type Status, StatusDot } from '@/components/ui/status-color-dot';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import type { SewageEvent } from '@/lib/models';
import { sewageEventStore } from '@/store';
import { useStore } from '@nanostores/react';

import {
  type Column,
  type ColumnDef,
  type SortingState,
  type VisibilityState,
  flexRender,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
} from '@tanstack/react-table';

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { useEffect, useState } from 'react';
import { formatDate, formatDuration, statusFn } from './lib';

import { Button } from '@/components/ui/button';
import { ArrowUpDown, Check, ChevronFirst, ChevronLast, ChevronLeft, ChevronRight, X as Cross } from 'lucide-react';

import { useBreakpoint } from '@/lib/hooks/useBreakpoint';
import { cn } from '@/lib/utils';

import { Skeleton } from '@/components/ui/skeleton';
import { useLazyStore } from '@/lib/useLazyStore';



const SortingButton = ({
  column,
  label,
  labelClassName = '',
}: { column: Column<SewageEvent, unknown>; label: string; labelClassName?: string }) => (
  <Button
    variant="ghost"
    onClick={() => column.toggleSorting(column.getIsSorted() === 'asc')}
  >
    <span className={cn('font-sm', labelClassName)}>
      {label}
    </span>
    <ArrowUpDown className="ml-2 h-4 w-4" />
  </Button>
);

const TooltipExplanation = ({
  label,
  explanation,
}: { label: string; explanation: string }) => (
  <TooltipProvider>
    <Tooltip>
      <TooltipTrigger>{label}</TooltipTrigger>
      <TooltipContent>{explanation}</TooltipContent>
    </Tooltip>
  </TooltipProvider>
);

const columns: ColumnDef<SewageEvent>[] = [
  {
    id: 'siteName',
    accessorKey: 'metadata.site_name',
    cell: ({ getValue }) => <span
      className='max-md:w-32 text-nowrap truncate inline-block'
    >{getValue() as string}</span>,
    enableSorting: false,
    header: () => (
      <TooltipExplanation
        label="Site Name"
        explanation="The name of the site as given by Northumbrian Water."
      />
    ),
  },
  // {
  //   id: 'eventType',
  //   accessorKey: 'event_type',
  //   cell: ({ getValue }) => {
  //     const eventType = getValue() as 'spill' | 'monitor offline';
  //     return (
  //       <div className="flex items-center">
  //         {eventType === 'spill' ? (
  //           <Outflow className='text-muted' />
  //         ) : (
  //           <Offline />
  //         )}
  //       </div>
  //     );
  //   },
  //   enableSorting: false,
  //   header: 'Event Type',
  // },
  {
    id: 'nearby',
    accessorKey: 'metadata.nearby',
    header: () => (
      <TooltipExplanation
        label="Nearby"
        explanation="On or just upstream of the rowable section of river in Durham City."
      />
    ),
    cell: ({ getValue }) => {
      const nearby = getValue() as boolean;
      return nearby ? (
        <Check className="text-red-500" />
      ) : (
        <Cross className="text-yellow-500" />
      );
    },
  },
  {
    id: 'eventStart',
    accessorKey: 'event_start',
    cell: ({ getValue }) => <span>{formatDate(getValue() as Date)}</span>,
    sortingFn: 'datetime',
    header: ({ column }) => (
      <SortingButton column={column} label="Start Date" />
    ),
  },
  {
    id: 'eventEnd',
    accessorKey: 'event_end',
    cell: ({ getValue }) => <span>{formatDate(getValue() as Date)}</span>,
    sortingFn: 'datetime',
    header: ({ column }) => <SortingButton column={column} label="End Date" />,
  },
  {
    id: 'duration',
    // accessorFn: (row) => row.event_end.getTime() - row.event_start.getTime(),
    accessorKey: 'event_duration_mins',
    cell: ({ getValue }) => <span>{formatDuration(getValue() as number)}</span>,
    header: 'Duration',
  },
  {
    id: 'status',
    accessorFn: (row) => statusFn(row),
    header: ({ column }) => <SortingButton column={column} label="Status" labelClassName='max-sm:hidden max-sm:sr-only' />,
    sortingFn: (rowA, rowB) =>
      (rowA.getValue('status') satisfies { status: Status }).status -
      (rowB.getValue('status') satisfies { status: Status }).status,
    cell: ({ getValue }) => {
      const { status, explanation } = getValue() as {
        status: Status;
        explanation: string;
      };
      return (
        <div className='flex items-center justify-center'>
          <TooltipProvider >
            <Tooltip>
              <TooltipTrigger>
                <StatusDot status={status} />
              </TooltipTrigger>
              <TooltipContent>
                <span>{explanation}</span>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      );
    },
  },
];

interface DataTableProps<TData, TValue> {
  columns: ColumnDef<TData, TValue>[];
  data: TData[];
  empty_message?: string;
  loading?: boolean;
}

function DataTable<TData, TValue>({
  columns,
  data,
  empty_message = 'No results.',
  loading = false,
}: DataTableProps<TData, TValue>) {
  const [sorting, setSorting] = useState<SortingState>([
    {
      id: 'status',
      desc: true,
    },
    {
      id: 'eventEnd',
      desc: true,
    },
  ]);

  const [pagination, setPagination] = useState({
    pageIndex: 0, //initial page index
    pageSize: 8, //default page size
  });

  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({});
  const { isAboveMd } = useBreakpoint('md');
  const { isAboveLg } = useBreakpoint('lg');

  useEffect(() => {
    setColumnVisibility({
      siteName: true,
      // eventType: true,
      nearby: isAboveLg,
      eventStart: isAboveMd,
      eventEnd: isAboveLg,
      duration: isAboveMd,
      status: true,
    });
  }, [isAboveMd, isAboveLg]);

  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    onSortingChange: setSorting,
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    onPaginationChange: setPagination,
    state: { sorting, columnVisibility, pagination },
  });

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
              {table.getRowModel().rows?.length ? (
                table.getRowModel().rows.map((row) => (
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
              ) : loading
                ? [...Array(pagination.pageSize)].map((_, index) => (
                  // biome-ignore lint/suspicious/noArrayIndexKey: Shouldn't need to update these at all
                  <TableRow key={index}>
                    {columns.map((column) => (
                      <TableCell key={column.id}>
                        <Skeleton className="h-2 w-full" />
                      </TableCell>
                    ))}
                  </TableRow>
                ))
                : <TableRow>
                  <TableCell
                    colSpan={columns.length}
                    className="h-24 text-center"
                  >
                    {empty_message}
                  </TableCell>
                </TableRow>}
            </TableBody>
          </Table>
        </div>
      </CardContent>
      <CardFooter>
        <div className="flex items-center justify-between w-full">
          <div className="text-sm text-muted">
            Showing Events {(pagination.pageIndex * pagination.pageSize) + 1} -{' '} {(pagination.pageIndex + 1) * pagination.pageSize} of{' '} {table.getRowCount()}
          </div>
          <div className="flex items-center gap-1">
            <Button
              variant="outline"
              size="icon"
              onClick={() => table.firstPage()}
              disabled={!table.getCanPreviousPage()}
            ><ChevronFirst className="h-4 w-4" /></Button>
            <Button
              variant="outline"
              size="icon"
              onClick={() => table.previousPage()}
              disabled={!table.getCanPreviousPage()}
            ><ChevronLeft className="h-4 w-4" /></Button>
            <Button
              variant="outline"
              size="icon"
              onClick={() => table.nextPage()}
              disabled={!table.getCanNextPage()}
            ><ChevronRight className="h-4 w-4" /></Button>
            <Button
              variant="outline"
              size="icon"
              onClick={() => table.lastPage()}
              disabled={!table.getCanNextPage()}
            ><ChevronLast className="h-4 w-4" /></Button>
          </div>
        </div>
      </CardFooter>
    </>
  );
}

export function SpillTable() {
  const { value, loading } = useLazyStore(sewageEventStore);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Storm Drain Outflows</CardTitle>
        <CardDescription>
          These events can indicate the spillage of sewage into the river. In
          this case, spillage should definitely not be lickage!
        </CardDescription>
      </CardHeader>
      <DataTable columns={columns} data={value || []} loading={loading} />
    </Card>
  );
}
