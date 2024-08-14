
import { Button } from '@/components/ui/button';
import type { SpillSite } from '@/types';
import type { PaginationState, Table } from '@tanstack/react-table';
import { ChevronFirst, ChevronLast, ChevronLeft, ChevronRight } from 'lucide-react';

type PaginationControlsProps = {
  table: Table<SpillSite>;
}

export function PaginationControls({ table }: PaginationControlsProps) {
  return (
    <div className="flex items-center gap-1">
      <Button
        variant="outline"
        size="icon"
        onClick={() => table.firstPage()}
        disabled={!table.getCanPreviousPage()}
      >
        <ChevronFirst className="h-4 w-4" />
      </Button>
      <Button
        variant="outline"
        size="icon"
        onClick={() => table.previousPage()}
        disabled={!table.getCanPreviousPage()}
      >
        <ChevronLeft className="h-4 w-4" />
      </Button>
      <Button
        variant="outline"
        size="icon"
        onClick={() => table.nextPage()}
        disabled={!table.getCanNextPage()}
      >
        <ChevronRight className="h-4 w-4" />
      </Button>
      <Button
        variant="outline"
        size="icon"
        onClick={() => table.lastPage()}
        disabled={!table.getCanNextPage()}
      >
        <ChevronLast className="h-4 w-4" />
      </Button>
    </div>
  );
}

type PaginationInfoProps = {
  table: Table<SpillSite>;
  pagination: PaginationState;
}

export function PaginationInfo({ table, pagination }: PaginationInfoProps) {
  return (
    <div className="text-sm text-muted-foreground">
      Showing Sites {(pagination.pageIndex * pagination.pageSize) + 1} -{' '} {(pagination.pageIndex + 1) * pagination.pageSize} of{' '} {table.getRowCount()}
    </div>
  );
}
