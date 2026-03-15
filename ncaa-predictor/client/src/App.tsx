import { useEffect, useState } from "react";
import { Link, Route, Switch, useLocation } from "wouter";
import { Router } from "wouter";
import { useHashLocation } from "wouter/use-hash-location";
import { QueryClientProvider } from "@tanstack/react-query";
import { BrainCircuit, CalendarRange, Moon, Sun, Trophy } from "lucide-react";
import { queryClient } from "./lib/queryClient";
import { Toaster } from "@/components/ui/toaster";
import MatchupPredictor from "./pages/matchup-predictor";
import DailySlatePage from "./pages/daily-slate";
import BracketSimulatorPage from "./pages/bracket-simulator";
import NotFound from "./pages/not-found";

function Navigation() {
  const [location] = useLocation();
  const links = [
    { href: "/", label: "Matchup", icon: BrainCircuit },
    { href: "/slate", label: "Slate", icon: CalendarRange },
    { href: "/bracket", label: "Bracket", icon: Trophy },
  ];

  return (
    <nav className="flex flex-wrap gap-2">
      {links.map((link) => {
        const Icon = link.icon;
        const isActive = location === link.href;

        return (
          <Link
            key={link.href}
            href={link.href}
            className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm transition-colors ${
              isActive
                ? "bg-primary text-primary-foreground shadow-sm"
                : "bg-muted/70 text-muted-foreground hover:text-foreground"
            }`}
          >
            <Icon className="h-4 w-4" />
            {link.label}
          </Link>
        );
      })}
    </nav>
  );
}

function AppRouter() {
  return (
    <Switch>
      <Route path="/" component={MatchupPredictor} />
      <Route path="/slate" component={DailySlatePage} />
      <Route path="/bracket" component={BracketSimulatorPage} />
      <Route component={NotFound} />
    </Switch>
  );
}

function Shell() {
  const [isDark, setIsDark] = useState(() =>
    window.matchMedia("(prefers-color-scheme: dark)").matches,
  );

  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDark);
  }, [isDark]);

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(249,115,22,0.08),_transparent_32%),radial-gradient(circle_at_bottom_right,_rgba(56,189,248,0.08),_transparent_24%),hsl(var(--background))]">
      <header className="sticky top-0 z-40 border-b border-border/70 bg-background/85 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-4 py-4 sm:px-6">
          <div>
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-primary text-primary-foreground shadow-lg shadow-primary/20">
                <BrainCircuit className="h-5 w-5" />
              </div>
              <div>
                <div className="text-lg font-semibold tracking-tight">NCAA Predictor Lab</div>
                <div className="text-xs uppercase tracking-[0.22em] text-muted-foreground">
                  API-backed men&apos;s D-I forecasting stack
                </div>
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <Navigation />
            <button
              type="button"
              onClick={() => setIsDark((value) => !value)}
              className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-border bg-card text-foreground transition-colors hover:border-primary/40"
              aria-label={`Switch to ${isDark ? "light" : "dark"} mode`}
            >
              {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
        <AppRouter />
      </main>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router hook={useHashLocation}>
        <Shell />
      </Router>
      <Toaster />
    </QueryClientProvider>
  );
}

export default App;
