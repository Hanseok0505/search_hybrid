// Fulltext index for document retrieval
CREATE FULLTEXT INDEX document_ft IF NOT EXISTS
FOR (d:Document)
ON EACH [d.title, d.content, d.summary, d.method_statement, d.risk_register];

// Optional lookups for top-down navigation
CREATE INDEX project_id_idx IF NOT EXISTS FOR (d:Document) ON (d.project_id);
CREATE INDEX package_code_idx IF NOT EXISTS FOR (d:Document) ON (d.package_code);
CREATE INDEX task_code_idx IF NOT EXISTS FOR (d:Document) ON (d.task_code);
CREATE INDEX wbs_code_idx IF NOT EXISTS FOR (d:Document) ON (d.wbs_code);
CREATE INDEX spec_section_idx IF NOT EXISTS FOR (d:Document) ON (d.spec_section);
