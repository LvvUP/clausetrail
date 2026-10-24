from clausetrail.structure.heading_features import LEVEL_BODY, numbering
from clausetrail.types import Block, BlockKind, SectionNode


def _style(block: Block) -> tuple:
    name, _ = numbering(block.text)
    return (name,) if name else ("plain", block.level)


# 各家条款的编号体系不同（平安的「1.」挂在「1.6」下面，国寿的「1.」是章），
# 所以层级按编号样式出现的先后来定：栈上已有的样式回到同级，新样式挂到当前标题下
def build_tree(blocks: list[Block]) -> SectionNode:
    root = SectionNode(title="", level=0, page=blocks[0].page if blocks else 1)
    stack: list[tuple[tuple, SectionNode]] = [((), root)]
    for block in blocks:
        if block.kind in (BlockKind.HEADER, BlockKind.FOOTER):
            continue
        if block.level > LEVEL_BODY and block.kind != BlockKind.TABLE:
            style = _style(block)
            depth = next((i for i, (s, _) in enumerate(stack) if s == style), None)
            if depth is not None:
                del stack[depth:]
            node = SectionNode(title=block.text.strip(), level=len(stack), page=block.page)
            stack[-1][1].children.append(node)
            stack.append((style, node))
        else:
            stack[-1][1].blocks.append(block)
    return root


def walk(node: SectionNode, ancestors: list[str] | None = None):
    ancestors = ancestors or []
    path = node.path(ancestors)
    yield node, path
    for child in node.children:
        yield from walk(child, path)


def outline(root: SectionNode, max_depth: int = 3) -> list[str]:
    return ["  " * (len(path) - 1) + path[-1] for node, path in walk(root) if path and len(path) <= max_depth]
