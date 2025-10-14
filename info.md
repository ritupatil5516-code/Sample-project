Smoke tests (happy path)
	1.	What’s my account status?
	2.	What’s my current balance?
	3.	What’s my available credit?
	4.	Show my last transaction.
	5.	Show my last payment.

Direct field (get_field) — account_summary
	6.	What is my credit limit?
	7.	What is my minimum payment due?
	8.	When is my next payment due date?
	9.	What’s the statement closing date for the latest cycle?
	10.	What is the APR for purchases?

Find latest (find_latest)
	11.	What was my latest posted transaction?
	12.	What was my latest payment and when did it post?
	13.	What was the latest statement period?

Sums & aggregates (sum_where / topk_by_sum)
	14.	How much did I spend in 2025-09?
	15.	How much did I spend this year?
	16.	How much did I spend in the last 12 months?
	17.	What’s the sum of refunds this year?
	18.	Top 5 merchants in the last 12 months.
	19.	Top categories this year.
	20.	Biggest single purchase last month.

Listing (list_where)
	21.	List all transactions from Apple in the last 12 months.
	22.	List declined transactions in 2025-09.
	23.	List reversed/chargeback transactions this year.
	24.	List payments made in 2025 Q3.
	25.	List purchases over $200 last month.

Semantic search over transactions (semantic_search)
	26.	Did I buy anything related to Apple?
	27.	Find travel-related purchases this year.
	28.	Any transactions for rideshare?
	29.	Anything that looks like subscriptions?
	30.	Any purchase notes mentioning gift or present?

Statements-focused
	31.	How much interest was charged in the latest statement?
	32.	Explain my interest breakdown for the last cycle.
	33.	What’s my trailing interest for the latest cycle?
	34.	Show the statement period with the highest interest charged.

Payments-focused
	35.	What was my largest payment this year?
	36.	Do I have any returned/failed payments?
	37.	What’s the average payment amount this year?

Policy/handbook RAG (knowledge only)
	38.	What is the late fee policy?
	39.	How does dispute/chargeback work and what’s the timeline?
	40.	How is interest calculated if I don’t pay in full?
	41.	What happens if I miss a payment?
	42.	How does cash advance interest/fees work?

Unified RAG (account + knowledge)
	43.	Why was I charged interest last month based on my usage and the policy?
	44.	I paid mid-cycle—why do I still see interest on the next statement?
	45.	Was my annual fee charged, and what does the policy say about it?
	46.	I see a foreign transaction—what are the fees and do I have any?
	47.	I returned an item—should I expect interest on that balance?

Time window variations
	48.	How much did I spend last month vs. the month before?
	49.	Compare Q2 vs. Q3 spending.
	50.	Show merchant totals for 2025-01.

Multi-turn memory (ask in sequence)
	51.	Q1: What’s my current balance?
Q2: How did it change compared to last statement?
Q3: Which transactions contributed most to the change?
	52.	Q1: Show Apple purchases last year.
Q2: Only show the top 3 by amount.
Q3: Sum those three.
	53.	Q1: Why was I charged interest last cycle?
Q2: If I pay in full now, will I have trailing interest next cycle?
Q3: What does the policy say about trailing interest?

Ambiguity & robustness
	54.	How much did I spend? (no period → system should choose latest or ask)
	55.	Show my largest purchase. (no period → latest/year-to-date)
	56.	Do I have any fees? (search transactions + policy context)
	57.	What’s my status? (should map to accountStatus)
	58.	Did I buy anything from “AMZ”? (abbrev/semantic)

Edge cases
	59.	List purchases over $5000 last month. (likely none)
	60.	Show reversed transactions in 2025-02. (may be none)
	61.	What’s my account owner name? (e.g., persons[0].name)
	62.	Show pending vs posted counts for last month.
	63.	What is my utilization? (current balance / credit limit)

Cross-account (if supported)
	64.	For ACCOUNT_ID A, what’s the current balance?
	65.	For ACCOUNT_ID B, show top merchants this year.
	66.	Compare interest across ACCOUNT_ID A vs B last quarter.

Stress & long-context
	67.	Summarize all spending trends over the last 12 months.
	68.	Generate a brief budget insight: where could I save next month?
	69.	Explain three possible reasons for interest charges given my activity, with sources.