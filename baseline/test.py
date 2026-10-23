from pathlib import Path

def get_frontmatter(filepath: Path):
    try:
        # importing the requirements
        import frontmatter
        from rich.console import Console
        from rich.markdown import Markdown
        
        console = Console()
        
        with open(filepath) as file:
            post = frontmatter.load(file)
        
        # title
        print(f"\nTitle: {post['title']}")
        
        # description
        print(f"\nDescription: {post["description"]}")
        
        # content
        md_content = Markdown("\n\n**Content**: " + str(post.content) + "\n\n")
        console.print(md_content)
        
        
        
    except Exception as e:
        print(f"Error: {e}")
        

if __name__ == "__main__":
    root = Path(__file__).parent
    md_file = root / "note.md"
    get_frontmatter(md_file)