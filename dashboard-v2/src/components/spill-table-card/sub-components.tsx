import { cn } from '@/lib/utils';
import { ArrowUpDown } from 'lucide-react';
import type { Column } from '@tanstack/react-table';
import type { SpillEvent } from '@/types';
import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

export const SortingButton = ({
  column,
  label,
  labelClassName = '',
}: { column: Column<SpillEvent, unknown>; label: string; labelClassName?: string }) => (
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

export const TooltipExplanation = ({
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
