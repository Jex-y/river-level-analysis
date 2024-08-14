import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
import type { SpillSite } from '@/types';
import type { Column } from '@tanstack/react-table';
import { ArrowUpDown } from 'lucide-react';

export const SortingButton = ({
  column,
  label,
  labelClassName = '',
}: { column: Column<SpillSite, unknown>; label: string; labelClassName?: string }) => (
  <Button
    className='p-0'
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
