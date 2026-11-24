"use strict";

const byId = (id) => document.getElementById(id);
const formatPercent = (value) => `${(Number(value) * 100).toFixed(1)}%`;

async function checkHealth() {
    const status = byId("service-status");
    if (!status) return;
    try {
        const response = await fetch(status.dataset.healthUrl, { headers: { Accept: "application/json" } });
        const payload = await response.json();
        const ready = response.ok && payload.ok;
        status.dataset.state = ready ? "ready" : "unavailable";
        status.querySelector(".status-text").textContent = ready ? "Model ready" : "Model unavailable";
    } catch {
        status.dataset.state = "unavailable";
        status.querySelector(".status-text").textContent = "Model unavailable";
    }
}

function initialiseClassifier() {
    const form = byId("prediction-form");
    if (!form) return;

    const title = byId("title");
    const article = byId("article");
    const counter = byId("character-count");
    const limitDisplay = counter.closest(".field-limit");
    const maximum = Number(form.dataset.maxCharacters);
    const submit = byId("submit-button");
    const clear = byId("clear-button");
    const message = byId("form-message");

    const updateCount = () => {
        const count = title.value.trim().length + article.value.trim().length;
        counter.textContent = count.toLocaleString("en-NZ");
        limitDisplay.classList.toggle("is-over", count > maximum);
        return count;
    };

    const setError = (text = "") => {
        message.textContent = text;
        message.hidden = !text;
    };

    const setLoading = (loading) => {
        submit.disabled = loading;
        submit.classList.toggle("is-loading", loading);
        submit.querySelector(".button-label").textContent = loading ? "Analysing" : "Check classification";
    };

    const showResult = (payload) => {
        const result = payload.prediction;
        const confidence = Number(result.confidence);
        byId("result-empty").hidden = true;
        byId("result-content").hidden = false;

        const badge = byId("result-badge");
        badge.textContent = result.classification;
        badge.classList.toggle("is-fake", result.decisive && Number(result.label) === 0);
        badge.classList.toggle("is-uncertain", !result.decisive);

        byId("confidence-value").textContent = formatPercent(confidence);
        byId("confidence-bar").style.width = formatPercent(confidence);
        byId("fake-probability").textContent = formatPercent(result.fake_probability);
        byId("real-probability").textContent = formatPercent(result.real_probability);
        byId("result-disclaimer").textContent = result.guidance || payload.disclaimer;

        const card = byId("result-card");
        const glow = !result.decisive
            ? "rgba(255,201,107,.18)"
            : Number(result.label) === 0
                ? "rgba(255,122,145,.2)"
                : "rgba(85,228,212,.18)";
        card.style.setProperty("--result-glow", glow);
        card.scrollIntoView({ behavior: "smooth", block: "nearest" });
    };

    title.addEventListener("input", updateCount);
    article.addEventListener("input", updateCount);

    clear.addEventListener("click", () => {
        form.reset();
        updateCount();
        setError();
        byId("result-content").hidden = true;
        byId("result-empty").hidden = false;
        title.focus();
    });

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        setError();
        const count = updateCount();
        if (count === 0) {
            setError("Enter a headline, article, or both before checking the classification.");
            title.focus();
            return;
        }
        if (count > maximum) {
            setError(`Keep the combined input below ${maximum.toLocaleString("en-NZ")} characters.`);
            return;
        }

        setLoading(true);
        try {
            const response = await fetch(form.action, {
                method: "POST",
                headers: { "Content-Type": "application/json", Accept: "application/json" },
                body: JSON.stringify({ title: title.value, article: article.value }),
            });
            const payload = await response.json();
            if (!response.ok || !payload.ok) {
                throw new Error(payload.error?.message || "The classification request could not be completed.");
            }
            showResult(payload);
        } catch (error) {
            setError(error.message || "The classification request could not be completed.");
        } finally {
            setLoading(false);
        }
    });

    updateCount();
}

document.addEventListener("DOMContentLoaded", () => {
    checkHealth();
    initialiseClassifier();
});
