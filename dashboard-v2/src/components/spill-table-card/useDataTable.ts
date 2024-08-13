import { useEffect, useState } from 'react';
import { useBreakpoint } from '@/hooks/useBreakpoint';

import {
  type ColumnDef,
  type SortingState,
  type VisibilityState,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
} from '@tanstack/react-table';

export function useDataTable<SpillEvent>(data: SpillEvent[], columns: ColumnDef<SpillEvent>[]) {
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

  return {
    table,
    pagination,
  }
}
