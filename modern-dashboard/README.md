## MeglerMonitor – Frontend

Moderne Next.js-klient for fase 2-dashboardet. Appen henter data fra FastAPI- backend (`backend/api.py`) og viser filtrerbar broker-oversikt med KPI-kort og kortbasert rangering.

### Kom i gang

1. Sørg for at API-et kjører:
   ```bash
   uvicorn backend.api:app --reload
   ```
2. Installer frontend-avhengigheter:
   ```bash
   npm install
   ```
3. Sett API-url (opprett `.env.local` ved behov):
   ```bash
   echo "NEXT_PUBLIC_MM_API=http://localhost:8000" > .env.local
   ```
4. Start Next.js i utviklingsmodus:
   ```bash
   npm run dev
   ```
5. Åpne [http://localhost:3000](http://localhost:3000). Filtrer venstresiden og trykk «Oppdater visning» for å hente nye resultater.

### Videre arbeid

- Legg til meglerdetaljside/modaler som treffer `/brokers/{broker_key}` for full profil, peers og anbefalinger.
- Pakk designet i et komponentbibliotek (Chakra, Material UI) eller legg til animasjoner med Framer Motion.
- Deploy Next.js til Vercel/Netlify og FastAPI til Railway/Render for delt demo.
