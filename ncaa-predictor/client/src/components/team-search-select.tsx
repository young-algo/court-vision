import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, Search, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { Team } from "@shared/schema";

interface TeamSearchSelectProps {
  label: string;
  teams: Team[];
  selectedTeam: Team | null;
  otherTeam: Team | null;
  onSelect: (team: Team) => void;
  side: "left" | "right";
}

export function TeamSearchSelect({
  label,
  teams,
  selectedTeam,
  otherTeam,
  onSelect,
  side,
}: TeamSearchSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState("");
  const rootRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (!rootRef.current?.contains(event.target as Node)) {
        setIsOpen(false);
        setSearch("");
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const filteredTeams = useMemo(() => {
    const query = search.trim().toLowerCase();
    return teams.filter((team) => {
      if (otherTeam?.id === team.id) {
        return false;
      }

      if (!query) {
        return true;
      }

      return (
        team.name.toLowerCase().includes(query) ||
        team.conference.toLowerCase().includes(query)
      );
    });
  }, [otherTeam, search, teams]);

  return (
    <div ref={rootRef} className="relative w-full">
      <label className="mb-1.5 block text-xs font-medium uppercase tracking-[0.24em] text-muted-foreground">
        {label}
      </label>
      <button
        data-testid={`team-select-${side}`}
        type="button"
        onClick={() => {
          setIsOpen((open) => !open);
          window.setTimeout(() => inputRef.current?.focus(), 40);
        }}
        className="flex w-full items-center justify-between gap-3 rounded-xl border border-border bg-card px-3 py-2.5 text-left transition-colors hover:border-primary/40"
      >
        {selectedTeam ? (
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold">{selectedTeam.shortName}</div>
            <div className="mt-1 flex items-center gap-2 text-[11px] text-muted-foreground">
              <span>{selectedTeam.conference}</span>
              <Badge variant="secondary" className="text-[10px]">
                #{selectedTeam.torvikRank}
              </Badge>
              <span>{selectedTeam.recordLabel}</span>
            </div>
          </div>
        ) : (
          <span className="text-sm text-muted-foreground">Select a team</span>
        )}
        <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground" />
      </button>

      {isOpen && (
        <div className="absolute z-50 mt-2 w-full overflow-hidden rounded-xl border border-border bg-popover shadow-xl">
          <div className="border-b border-border p-2">
            <div className="relative">
              <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
              <input
                ref={inputRef}
                data-testid={`team-search-${side}`}
                type="search"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search by team or conference"
                className="w-full rounded-lg bg-transparent py-1.5 pl-8 pr-8 text-sm outline-none placeholder:text-muted-foreground"
              />
              {search ? (
                <button
                  type="button"
                  onClick={() => setSearch("")}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground"
                >
                  <X className="h-3 w-3" />
                </button>
              ) : null}
            </div>
          </div>
          <div className="max-h-72 overflow-y-auto">
            {filteredTeams.map((team) => (
              <button
                key={team.id}
                type="button"
                data-testid={`team-option-${team.id}`}
                onClick={() => {
                  onSelect(team);
                  setIsOpen(false);
                  setSearch("");
                }}
                className="flex w-full items-center justify-between px-3 py-2.5 text-left transition-colors hover:bg-accent/50"
              >
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium">{team.shortName}</div>
                  <div className="mt-0.5 text-[11px] text-muted-foreground">
                    {team.conference}
                  </div>
                </div>
                <div className="ml-3 shrink-0 text-right text-[11px] text-muted-foreground">
                  <div>#{team.torvikRank}</div>
                  <div>{team.recordLabel}</div>
                </div>
              </button>
            ))}
            {filteredTeams.length === 0 ? (
              <div className="px-3 py-6 text-center text-sm text-muted-foreground">
                No teams match that search.
              </div>
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
}
