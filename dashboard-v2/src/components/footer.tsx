
import { ModeToggle } from '@/components/theme-settings';

export function Footer() {
  return (<footer className="py-6 md:py-0">
    <div
      className="container flex flex-col items-center justify-between gap-4 md:h-24 md:flex-row"
    >
      <ModeToggle />
      <p
        className="text-balance text-center text-sm leading-loose text-muted-foreground w-full"
      >
        <span>Built by Ed Jex.</span>
        {" "}
        <span>
          Issues / Feature requests: {" "}
          <a
            href="https://github.com/Jex-y/river-level-analysis/issues"
            target="_blank"
            rel="noreferrer"
            className="font-medium underline underline-offset-4"
          >
            GitHub
          </a>
        </span>
      </p>
    </div>
  </footer>
  )
}