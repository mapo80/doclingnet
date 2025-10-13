# Agent instructions

## Installazione .NET
1. Rendi eseguibile lo script `donetinstall` presente nella root del repository:
   ```bash
   chmod +x donetinstall
   ```
2. Installa .NET 8 utilizzando la cartella `$HOME/dotnet` come destinazione (aggiorna la variabile PATH se necessario):
   ```bash
   ./donetinstall --channel 8.0 --install-dir "$HOME/dotnet"
   export PATH="$HOME/dotnet:$PATH"
   ```
3. Verifica l'installazione controllando la versione:
   ```bash
   dotnet --version
   ```
4. Ogni nuova sessione shell richiede di riesportare la variabile PATH (passo 2) prima di invocare `dotnet`.

5. Esegui i test del repository con la solution `TableFormerSdk.sln`:
   ```bash
   dotnet test TableFormerSdk.sln
   ```
