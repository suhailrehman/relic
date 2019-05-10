--
-- PostgreSQL database dump
--

-- Dumped from database version 11.2 (Ubuntu 11.2-1.pgdg18.04+1)
-- Dumped by pg_dump version 11.2 (Ubuntu 11.2-1.pgdg18.04+1)

-- Started on 2019-05-10 18:25:35 CDT

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 8 (class 2615 OID 16514)
-- Name: lineage; Type: SCHEMA; Schema: -; Owner: lineage
--

CREATE SCHEMA lineage;


ALTER SCHEMA lineage OWNER TO lineage;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- TOC entry 199 (class 1259 OID 16528)
-- Name: artifact; Type: TABLE; Schema: lineage; Owner: lineage
--

CREATE TABLE lineage.artifact (
    id uuid NOT NULL,
    filename text NOT NULL,
    path text,
    workflow_id uuid NOT NULL
);


ALTER TABLE lineage.artifact OWNER TO lineage;

--
-- TOC entry 200 (class 1259 OID 16538)
-- Name: artifact_column; Type: TABLE; Schema: lineage; Owner: suhail
--

CREATE TABLE lineage.artifact_column (
    id uuid NOT NULL,
    column_name character varying(45) NOT NULL,
    column_type character varying(45),
    sketch_signature character varying(45),
    artifact_id uuid NOT NULL,
    sketch_time integer
);


ALTER TABLE lineage.artifact_column OWNER TO suhail;

--
-- TOC entry 202 (class 1259 OID 16568)
-- Name: cluster; Type: TABLE; Schema: lineage; Owner: suhail
--

CREATE TABLE lineage.cluster (
    id uuid NOT NULL,
    artifact_id uuid NOT NULL,
    type character varying(45),
    experiment_id uuid NOT NULL
);


ALTER TABLE lineage.cluster OWNER TO suhail;

--
-- TOC entry 197 (class 1259 OID 16515)
-- Name: experiment; Type: TABLE; Schema: lineage; Owner: suhail
--

CREATE TABLE lineage.experiment (
    id uuid NOT NULL,
    create_time character varying(45) NOT NULL,
    start_time timestamp without time zone NOT NULL,
    parameters json,
    commit_hash character varying(2000)
);


ALTER TABLE lineage.experiment OWNER TO suhail;

--
-- TOC entry 201 (class 1259 OID 16548)
-- Name: ground_truth_edge; Type: TABLE; Schema: lineage; Owner: suhail
--

CREATE TABLE lineage.ground_truth_edge (
    id uuid NOT NULL,
    artifact_1 uuid,
    artifact_2 uuid,
    workflow_id uuid NOT NULL
);


ALTER TABLE lineage.ground_truth_edge OWNER TO suhail;

--
-- TOC entry 203 (class 1259 OID 16583)
-- Name: relationship_edge; Type: TABLE; Schema: lineage; Owner: suhail
--

CREATE TABLE lineage.relationship_edge (
    id uuid NOT NULL,
    artifact_1 uuid NOT NULL,
    artifact_2 uuid NOT NULL,
    distance_type character varying(45),
    distance_value double precision,
    experiment_id uuid NOT NULL
);


ALTER TABLE lineage.relationship_edge OWNER TO suhail;

--
-- TOC entry 204 (class 1259 OID 16603)
-- Name: time_log; Type: TABLE; Schema: lineage; Owner: suhail
--

CREATE TABLE lineage.time_log (
    task_name character varying(45),
    time_taken integer,
    experiment_id uuid
);


ALTER TABLE lineage.time_log OWNER TO suhail;

--
-- TOC entry 198 (class 1259 OID 16523)
-- Name: workflow; Type: TABLE; Schema: lineage; Owner: suhail
--

CREATE TABLE lineage.workflow (
    id uuid NOT NULL,
    directory_path text NOT NULL
);


ALTER TABLE lineage.workflow OWNER TO suhail;

--
-- TOC entry 2840 (class 2606 OID 16542)
-- Name: artifact_column artifact_column_pkey; Type: CONSTRAINT; Schema: lineage; Owner: suhail
--

ALTER TABLE ONLY lineage.artifact_column
    ADD CONSTRAINT artifact_column_pkey PRIMARY KEY (id);


--
-- TOC entry 2836 (class 2606 OID 16532)
-- Name: artifact artifact_pkey; Type: CONSTRAINT; Schema: lineage; Owner: lineage
--

ALTER TABLE ONLY lineage.artifact
    ADD CONSTRAINT artifact_pkey PRIMARY KEY (id);


--
-- TOC entry 2844 (class 2606 OID 16572)
-- Name: cluster cluster_pkey; Type: CONSTRAINT; Schema: lineage; Owner: suhail
--

ALTER TABLE ONLY lineage.cluster
    ADD CONSTRAINT cluster_pkey PRIMARY KEY (id);


--
-- TOC entry 2832 (class 2606 OID 16615)
-- Name: workflow directory_path_uq; Type: CONSTRAINT; Schema: lineage; Owner: suhail
--

ALTER TABLE ONLY lineage.workflow
    ADD CONSTRAINT directory_path_uq UNIQUE (directory_path);


--
-- TOC entry 2830 (class 2606 OID 16522)
-- Name: experiment experiment_pkey; Type: CONSTRAINT; Schema: lineage; Owner: suhail
--

ALTER TABLE ONLY lineage.experiment
    ADD CONSTRAINT experiment_pkey PRIMARY KEY (id);


--
-- TOC entry 2842 (class 2606 OID 16552)
-- Name: ground_truth_edge ground_truth_edge_pkey; Type: CONSTRAINT; Schema: lineage; Owner: suhail
--

ALTER TABLE ONLY lineage.ground_truth_edge
    ADD CONSTRAINT ground_truth_edge_pkey PRIMARY KEY (id);


--
-- TOC entry 2846 (class 2606 OID 16587)
-- Name: relationship_edge relationship_edge_pkey; Type: CONSTRAINT; Schema: lineage; Owner: suhail
--

ALTER TABLE ONLY lineage.relationship_edge
    ADD CONSTRAINT relationship_edge_pkey PRIMARY KEY (id);


--
-- TOC entry 2838 (class 2606 OID 16688)
-- Name: artifact unique_artifact_constraint; Type: CONSTRAINT; Schema: lineage; Owner: lineage
--

ALTER TABLE ONLY lineage.artifact
    ADD CONSTRAINT unique_artifact_constraint UNIQUE (filename, path);


--
-- TOC entry 2834 (class 2606 OID 16527)
-- Name: workflow workflow_pkey; Type: CONSTRAINT; Schema: lineage; Owner: suhail
--

ALTER TABLE ONLY lineage.workflow
    ADD CONSTRAINT workflow_pkey PRIMARY KEY (id);


--
-- TOC entry 2848 (class 2606 OID 16543)
-- Name: artifact_column fk_artifact_column_artifact1; Type: FK CONSTRAINT; Schema: lineage; Owner: suhail
--

ALTER TABLE ONLY lineage.artifact_column
    ADD CONSTRAINT fk_artifact_column_artifact1 FOREIGN KEY (artifact_id) REFERENCES lineage.artifact(id) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 2847 (class 2606 OID 16533)
-- Name: artifact fk_artifact_workflow; Type: FK CONSTRAINT; Schema: lineage; Owner: lineage
--

ALTER TABLE ONLY lineage.artifact
    ADD CONSTRAINT fk_artifact_workflow FOREIGN KEY (workflow_id) REFERENCES lineage.workflow(id) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 2852 (class 2606 OID 16573)
-- Name: cluster fk_cluster_artifact; Type: FK CONSTRAINT; Schema: lineage; Owner: suhail
--

ALTER TABLE ONLY lineage.cluster
    ADD CONSTRAINT fk_cluster_artifact FOREIGN KEY (artifact_id) REFERENCES lineage.artifact(id) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 2853 (class 2606 OID 16578)
-- Name: cluster fk_cluster_experiment; Type: FK CONSTRAINT; Schema: lineage; Owner: suhail
--

ALTER TABLE ONLY lineage.cluster
    ADD CONSTRAINT fk_cluster_experiment FOREIGN KEY (experiment_id) REFERENCES lineage.experiment(id);


--
-- TOC entry 2854 (class 2606 OID 16588)
-- Name: relationship_edge fk_experiment; Type: FK CONSTRAINT; Schema: lineage; Owner: suhail
--

ALTER TABLE ONLY lineage.relationship_edge
    ADD CONSTRAINT fk_experiment FOREIGN KEY (experiment_id) REFERENCES lineage.experiment(id);


--
-- TOC entry 2849 (class 2606 OID 16553)
-- Name: ground_truth_edge fk_ground_truth_edge_artifact1; Type: FK CONSTRAINT; Schema: lineage; Owner: suhail
--

ALTER TABLE ONLY lineage.ground_truth_edge
    ADD CONSTRAINT fk_ground_truth_edge_artifact1 FOREIGN KEY (artifact_1) REFERENCES lineage.artifact(id) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 2850 (class 2606 OID 16558)
-- Name: ground_truth_edge fk_ground_truth_edge_artifact2; Type: FK CONSTRAINT; Schema: lineage; Owner: suhail
--

ALTER TABLE ONLY lineage.ground_truth_edge
    ADD CONSTRAINT fk_ground_truth_edge_artifact2 FOREIGN KEY (artifact_1) REFERENCES lineage.artifact(id) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 2851 (class 2606 OID 16563)
-- Name: ground_truth_edge fk_ground_truth_edge_workflow; Type: FK CONSTRAINT; Schema: lineage; Owner: suhail
--

ALTER TABLE ONLY lineage.ground_truth_edge
    ADD CONSTRAINT fk_ground_truth_edge_workflow FOREIGN KEY (workflow_id) REFERENCES lineage.workflow(id) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 2855 (class 2606 OID 16593)
-- Name: relationship_edge fk_relationship_edge_artifact1; Type: FK CONSTRAINT; Schema: lineage; Owner: suhail
--

ALTER TABLE ONLY lineage.relationship_edge
    ADD CONSTRAINT fk_relationship_edge_artifact1 FOREIGN KEY (artifact_1) REFERENCES lineage.artifact(id) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 2856 (class 2606 OID 16598)
-- Name: relationship_edge fk_relationship_edge_artifact2; Type: FK CONSTRAINT; Schema: lineage; Owner: suhail
--

ALTER TABLE ONLY lineage.relationship_edge
    ADD CONSTRAINT fk_relationship_edge_artifact2 FOREIGN KEY (artifact_2) REFERENCES lineage.artifact(id) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 2857 (class 2606 OID 16606)
-- Name: time_log fk_timelog_experiment1; Type: FK CONSTRAINT; Schema: lineage; Owner: suhail
--

ALTER TABLE ONLY lineage.time_log
    ADD CONSTRAINT fk_timelog_experiment1 FOREIGN KEY (experiment_id) REFERENCES lineage.experiment(id);


--
-- TOC entry 2984 (class 0 OID 0)
-- Dependencies: 198
-- Name: TABLE workflow; Type: ACL; Schema: lineage; Owner: suhail
--

GRANT ALL ON TABLE lineage.workflow TO lineage;


-- Completed on 2019-05-10 18:25:35 CDT

--
-- PostgreSQL database dump complete
--
