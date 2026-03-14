-- Add Foreign Key Constraints to fact_sales table
-- Run this in pgAdmin to create relationships

BEGIN;

-- FK to dim_date
ALTER TABLE public.fact_sales
ADD CONSTRAINT fk_fact_sales_date
FOREIGN KEY (date_key) 
REFERENCES public.dim_date(date_key);

-- FK to dim_store
ALTER TABLE public.fact_sales
ADD CONSTRAINT fk_fact_sales_store
FOREIGN KEY (store_key) 
REFERENCES public.dim_store(store_key);

-- FK to dim_product
ALTER TABLE public.fact_sales
ADD CONSTRAINT fk_fact_sales_product
FOREIGN KEY (product_key) 
REFERENCES public.dim_product(product_key);

COMMIT;

-- Verify constraints
SELECT 
    conname as constraint_name,
    conrelid::regclass as table_name,
    confrelid::regclass as references_table
FROM pg_constraint
WHERE contype = 'f' 
  AND conrelid::regclass::text = 'fact_sales';
