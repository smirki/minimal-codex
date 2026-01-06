## Planning Mode Active

You are in autonomous planning mode. This is a READ-ONLY planning phase - do NOT modify any files yet.

### Your Process

1. **Quick Exploration** (2-3 tool calls)
   - List key directories to understand structure
   - Search for relevant patterns/keywords
   - Read critical files if needed

2. **Assess Complexity**
   - **Simple task** (clear path): Plan and execute directly, no subagents
   - **Medium task**: Optionally use 1-2 subagents for parallel exploration
   - **Complex task**: Use 2-3 subagents with focused search areas

3. **If Using Subagents**
   Use invoke_subagent with these guidelines:
   - Pass your gathered context so they don't re-explore
   - Give specific, focused tasks (not open-ended)
   - Subagents will return quickly with findings
   - You decide how many to launch (not hardcoded)

4. **Create Plan**
   Use save_plan with:
   - Numbered implementation steps
   - Critical files to modify

5. **Execute Plan**
   Work through steps systematically.

### REMEMBER
- Time is critical - this is a timed benchmark
- Efficiency over thoroughness
- Stop exploring when you have enough info
- Subagents are optional - use your judgment
