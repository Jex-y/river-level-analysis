
import type { Table, PaginationState } from '@tanstack/react-table';
import type { SpillEvent } from '@/types';
import { Button } from '@/components/ui/button';
import { ChevronFirst, ChevronLast, ChevronLeft, ChevronRight } from 'lucide-react';

type PaginationControlsProps = {
  table: Table<SpillEvent>;
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
  table: Table<SpillEvent>;
  pagination: PaginationState;
}

export function PaginationInfo({ table, pagination }: PaginationInfoProps) {
  return (
    <div className="text-sm text-muted-foreground">
      Showing Events {(pagination.pageIndex * pagination.pageSize) + 1} -{' '} {(pagination.pageIndex + 1) * pagination.pageSize} of{' '} {table.getRowCount()}
    </div>
  );
}
