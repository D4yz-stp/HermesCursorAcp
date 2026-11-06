# cursor-acp

Provider Hermes para o processo `agent acp`. A conta é a que já estiver autenticada nesse `agent`.

## Instalação

Copiar o conteúdo de `cursor_acp/` para `$HERMES_HOME/plugins/model-providers/cursor-acp/`. O directório de destino fica com `plugin.yaml` e `__init__.py` na raiz:

```bash
mkdir -p "$HERMES_HOME/plugins/model-providers/cursor-acp"
cp -a cursor_acp/. "$HERMES_HOME/plugins/model-providers/cursor-acp/"
```

O Hermes arranca o processo com o comando `agent acp`.

## Modelo

O id pedido tem de figurar na lista que a sessão ACP anuncia. Um id fora dessa lista é um erro, e a mensagem mostra a lista anunciada. Nesse caso o cliente não envia `session/prompt`.

## Usage

Cada completion traz o atributo `usage` com o valor `None`.

Com `auxiliary.<task>.provider` em `auto`, a completion auxiliar constrói o mesmo cliente e lança outro `agent acp`. A revisão de fundo é a excepção: `SUPPORTS_HERMES_TOOL_CALLS` é `False`, o fork não se constrói, e o plugin não escreve outro `auxiliary.background_review`. O plugin não declara `supports_vision`.

## Limites da cerca

O processo corre com cwd, `HOME`, `CURSOR_CONFIG_DIR` e `CURSOR_DATA_DIR` privados. A primeira versão deixa estes pontos por fechar:

- Os comandos globais do CLI continuam anunciados em `available_commands_update`.
- O processo pode copiar o `cli-config.json` da conta para o directório privado. A cópia fica nesse directório, com as chaves da conta, e não volta para o autor.
- O ficheiro de login é `auth.json` em `$XDG_CONFIG_HOME/cursor/auth.json`. Sem `XDG_CONFIG_HOME`, o caminho é `~/.config/cursor/auth.json`.
- As user rules da conta podem aparecer no store da sessão.
- Um resultado vazio de Grep ou de Glob não prova que o ficheiro não foi aberto.
