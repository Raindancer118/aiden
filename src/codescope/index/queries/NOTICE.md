# Tags query provenance

Most `*-tags.scm` files in this directory are vendored from the
[Aider](https://github.com/Aider-AI/aider) project
(`aider/queries/tree-sitter-language-pack/`), which is licensed under the
**Apache License 2.0**. They use the standardized tree-sitter "tags" capture
vocabulary (`@name.definition.*`, `@definition.*`, `@name.reference.*`).

Copyright belongs to the Aider authors; see
<https://github.com/Aider-AI/aider/blob/main/LICENSE.txt>.

The following files are **original to Volantic Codescope** (also provided under
Apache-2.0 for consistency):

- `typescript-tags.scm`
- `tsx-tags.scm`

The file names match `tree-sitter-language-pack` language names so they can be
loaded generically by `codescope/index/parser.py`.
