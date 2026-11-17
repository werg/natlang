var __defProp = Object.defineProperty;
var __defProps = Object.defineProperties;
var __getOwnPropDescs = Object.getOwnPropertyDescriptors;
var __getOwnPropSymbols = Object.getOwnPropertySymbols;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __propIsEnum = Object.prototype.propertyIsEnumerable;
var __knownSymbol = (name, symbol) => (symbol = Symbol[name]) ? symbol : Symbol.for("Symbol." + name);
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __spreadValues = (a, b) => {
  for (var prop in b || (b = {}))
    if (__hasOwnProp.call(b, prop))
      __defNormalProp(a, prop, b[prop]);
  if (__getOwnPropSymbols)
    for (var prop of __getOwnPropSymbols(b)) {
      if (__propIsEnum.call(b, prop))
        __defNormalProp(a, prop, b[prop]);
    }
  return a;
};
var __spreadProps = (a, b) => __defProps(a, __getOwnPropDescs(b));
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
var __async = (__this, __arguments, generator) => {
  return new Promise((resolve, reject) => {
    var fulfilled = (value) => {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    };
    var rejected = (value) => {
      try {
        step(generator.throw(value));
      } catch (e) {
        reject(e);
      }
    };
    var step = (x) => x.done ? resolve(x.value) : Promise.resolve(x.value).then(fulfilled, rejected);
    step((generator = generator.apply(__this, __arguments)).next());
  });
};
var __await = function(promise, isYieldStar) {
  this[0] = promise;
  this[1] = isYieldStar;
};
var __asyncGenerator = (__this, __arguments, generator) => {
  var resume = (k, v, yes, no) => {
    try {
      var x = generator[k](v), isAwait = (v = x.value) instanceof __await, done = x.done;
      Promise.resolve(isAwait ? v[0] : v).then((y) => isAwait ? resume(k === "return" ? k : "next", v[1] ? { done: y.done, value: y.value } : y, yes, no) : yes({ value: y, done })).catch((e) => resume("throw", e, yes, no));
    } catch (e) {
      no(e);
    }
  }, method = (k) => it[k] = (x) => new Promise((yes, no) => resume(k, x, yes, no)), it = {};
  return generator = generator.apply(__this, __arguments), it[__knownSymbol("asyncIterator")] = () => it, method("next"), method("throw"), method("return"), it;
};
var __forAwait = (obj, it, method) => (it = obj[__knownSymbol("asyncIterator")]) ? it.call(obj) : (obj = obj[__knownSymbol("iterator")](), it = {}, method = (key, fn) => (fn = obj[key]) && (it[key] = (arg) => new Promise((yes, no, done) => (arg = fn.call(obj, arg), done = arg.done, Promise.resolve(arg.value).then((value) => yes({ value, done }), no)))), method("next"), method("return"), it);

// src/glue/messages.ts
var GLUE_VERSION = 1;
var GLUE_MESSAGE_PROTOTYPES = {
  "erro_evt": {
    "name": "erro_evt",
    "structName": "glue_msg_error",
    "className": "GlueMsgError",
    "fields": [
      {
        "type": "str",
        "name": "message",
        "isNullable": false
      }
    ]
  },
  "load_req": {
    "name": "load_req",
    "structName": "glue_msg_load_req",
    "className": "GlueMsgLoadReq",
    "fields": [
      {
        "type": "arr_str",
        "name": "model_paths",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "mmproj_path",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "n_ctx_auto",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "use_mmap",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "use_mlock",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_gpu_layers",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_threads",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "model_alias",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "log_level",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "embeddings",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "offload_kqv",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_batch",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_ubatch",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_parallel",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "pooling_type",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "rope_scaling_type",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "rope_freq_base",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "rope_freq_scale",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_ext_factor",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_attn_factor",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_beta_fast",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_beta_slow",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "yarn_orig_ctx",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "cache_type_k",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "cache_type_v",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "kv_unified",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "flash_attn",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "swa_full",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_ctx_checkpoints",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "checkpoint_min_step",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "chat_template",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "jinja",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "default_template_kwargs_keys",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "default_template_kwargs_vals",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "reasoning",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "image_min_tokens",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "image_max_tokens",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "warmup",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "no_kv_offload",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "mmproj_offload",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "cont_batching",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_keep",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "ctx_shift",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "cache_idle_slots",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_cache_reuse",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "lora_paths",
        "isNullable": true
      },
      {
        "type": "arr_float",
        "name": "lora_scales",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "lora_init_without_apply",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "native_state_paths",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "spec_draft_model",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_ngl",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_n_max",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_n_min",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "spec_draft_p_min",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_threads",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_threads_batch",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "kv_overrides_keys",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "kv_overrides_vals",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "reasoning_budget_tokens",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "reasoning_budget_message",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "reasoning_format",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "skip_chat_parsing",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "prefill_assistant",
        "isNullable": true
      }
    ]
  },
  "load_res": {
    "name": "load_res",
    "structName": "glue_msg_load_res",
    "className": "GlueMsgLoadRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_batch",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ubatch",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_vocab",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx_train",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_embd",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_layer",
        "isNullable": false
      },
      {
        "type": "arr_str",
        "name": "metadata_key",
        "isNullable": false
      },
      {
        "type": "arr_str",
        "name": "metadata_val",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_bos",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_eos",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_eot",
        "isNullable": false
      },
      {
        "type": "arr_int",
        "name": "list_tokens_eog",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "add_bos_token",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "add_eos_token",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_encoder",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_decoder_start",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "media_marker",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_image_input",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_audio_input",
        "isNullable": false
      }
    ]
  },
  "cmpl_req": {
    "name": "cmpl_req",
    "structName": "glue_msg_completion_req",
    "className": "GlueMsgCompletionReq",
    "fields": [
      {
        "type": "bool",
        "name": "is_chat",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      },
      {
        "type": "arr_raw",
        "name": "files",
        "isNullable": false
      }
    ]
  },
  "cmpl_res": {
    "name": "cmpl_res",
    "structName": "glue_msg_completion_res",
    "className": "GlueMsgCompletionRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "embd_req": {
    "name": "embd_req",
    "structName": "glue_msg_embedding_req",
    "className": "GlueMsgEmbeddingReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      },
      {
        "type": "arr_raw",
        "name": "files",
        "isNullable": false
      }
    ]
  },
  "embd_res": {
    "name": "embd_res",
    "structName": "glue_msg_embedding_res",
    "className": "GlueMsgEmbeddingRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "rrnk_req": {
    "name": "rrnk_req",
    "structName": "glue_msg_rerank_req",
    "className": "GlueMsgRerankReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "rrnk_res": {
    "name": "rrnk_res",
    "structName": "glue_msg_rerank_res",
    "className": "GlueMsgRerankRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "sys1_req": {
    "name": "sys1_req",
    "structName": "glue_msg_systemone_req",
    "className": "GlueMsgSystemoneReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "sys1_res": {
    "name": "sys1_res",
    "structName": "glue_msg_systemone_res",
    "className": "GlueMsgSystemoneRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "scor_req": {
    "name": "scor_req",
    "structName": "glue_msg_score_req",
    "className": "GlueMsgScoreReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "scor_res": {
    "name": "scor_res",
    "structName": "glue_msg_score_res",
    "className": "GlueMsgScoreRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "slot_req": {
    "name": "slot_req",
    "structName": "glue_msg_slots_req",
    "className": "GlueMsgSlotsReq",
    "fields": [
      {
        "type": "int",
        "name": "id_slot",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "action",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "filename",
        "isNullable": false
      }
    ]
  },
  "slot_res": {
    "name": "slot_res",
    "structName": "glue_msg_slots_res",
    "className": "GlueMsgSlotsRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "nsta_req": {
    "name": "nsta_req",
    "structName": "glue_msg_native_state_req",
    "className": "GlueMsgNativeStateReq",
    "fields": [
      {
        "type": "str",
        "name": "action",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "id",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "filename",
        "isNullable": false
      }
    ]
  },
  "nsta_res": {
    "name": "nsta_res",
    "structName": "glue_msg_native_state_res",
    "className": "GlueMsgNativeStateRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "meta_json",
        "isNullable": false
      }
    ]
  },
  "tmpl_req": {
    "name": "tmpl_req",
    "structName": "glue_msg_apply_template_req",
    "className": "GlueMsgApplyTemplateReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "tmpl_res": {
    "name": "tmpl_res",
    "structName": "glue_msg_apply_template_res",
    "className": "GlueMsgApplyTemplateRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "prompt",
        "isNullable": false
      }
    ]
  },
  "gres_req": {
    "name": "gres_req",
    "structName": "glue_msg_get_result_req",
    "className": "GlueMsgGetResultReq",
    "fields": [
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "gres_res": {
    "name": "gres_res",
    "structName": "glue_msg_get_result_res",
    "className": "GlueMsgGetResultRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_more",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "is_error",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "cncl_req": {
    "name": "cncl_req",
    "structName": "glue_msg_cancel_req",
    "className": "GlueMsgCancelReq",
    "fields": [
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "cncl_res": {
    "name": "cncl_res",
    "structName": "glue_msg_cancel_res",
    "className": "GlueMsgCancelRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      }
    ]
  },
  "tbop_req": {
    "name": "tbop_req",
    "structName": "glue_msg_test_backend_ops_req",
    "className": "GlueMsgTestBackendOpsReq",
    "fields": [
      {
        "type": "arr_str",
        "name": "args",
        "isNullable": false
      }
    ]
  },
  "tbop_res": {
    "name": "tbop_res",
    "structName": "glue_msg_test_backend_ops_res",
    "className": "GlueMsgTestBackendOpsRes",
    "fields": [
      {
        "type": "int",
        "name": "retcode",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      }
    ]
  }
};

// src/glue/glue.ts
var GLUE_MAGIC = new Uint8Array([71, 76, 85, 69]);
var GLUE_DTYPE_NULL = 0;
var GLUE_DTYPE_BOOL = 1;
var GLUE_DTYPE_INT = 2;
var GLUE_DTYPE_FLOAT = 3;
var GLUE_DTYPE_STRING = 4;
var GLUE_DTYPE_RAW = 5;
var GLUE_DTYPE_ARRAY_BOOL = 6;
var GLUE_DTYPE_ARRAY_INT = 7;
var GLUE_DTYPE_ARRAY_FLOAT = 8;
var GLUE_DTYPE_ARRAY_STRING = 9;
var GLUE_DTYPE_ARRAY_RAW = 10;
var TYPE_MAP = {
  str: GLUE_DTYPE_STRING,
  int: GLUE_DTYPE_INT,
  float: GLUE_DTYPE_FLOAT,
  bool: GLUE_DTYPE_BOOL,
  raw: GLUE_DTYPE_RAW,
  arr_str: GLUE_DTYPE_ARRAY_STRING,
  arr_int: GLUE_DTYPE_ARRAY_INT,
  arr_float: GLUE_DTYPE_ARRAY_FLOAT,
  arr_bool: GLUE_DTYPE_ARRAY_BOOL,
  arr_raw: GLUE_DTYPE_ARRAY_RAW,
  null: GLUE_DTYPE_NULL
};
function glueDeserialize(buf) {
  let offset = 0;
  const view = new DataView(buf.buffer);
  const readUint32 = () => {
    const value = view.getUint32(offset, true);
    offset += 4;
    return value;
  };
  const readInt32 = () => {
    const value = view.getInt32(offset, true);
    offset += 4;
    return value;
  };
  const readFloat = () => {
    const value = view.getFloat32(offset, true);
    offset += 4;
    return value;
  };
  const readBool = () => {
    return readUint32() !== 0;
  };
  const readString = (customLen) => {
    const length = customLen != null ? customLen : readUint32();
    const value = new TextDecoder().decode(buf.slice(offset, offset + length));
    offset += length;
    return value;
  };
  const readRaw = () => {
    const length = readUint32();
    const value = buf.slice(offset, offset + length);
    offset += length;
    return value;
  };
  const readArray = (readItem) => {
    const length = readUint32();
    const value = new Array(length);
    for (let i = 0; i < length; i++) {
      value[i] = readItem();
    }
    return value;
  };
  const readNull = () => null;
  const readField = (field) => {
    switch (field.type) {
      case "str":
        return readString();
      case "int":
        return readInt32();
      case "float":
        return readFloat();
      case "bool":
        return readBool();
      case "raw":
        return readRaw();
      case "arr_str":
        return readArray(readString);
      case "arr_int":
        return readArray(readInt32);
      case "arr_float":
        return readArray(readFloat);
      case "arr_bool":
        return readArray(readBool);
      case "arr_raw":
        return readArray(readRaw);
      case "null":
        return readNull();
    }
  };
  const magicValid = buf[0] === GLUE_MAGIC[0] && buf[1] === GLUE_MAGIC[1] && buf[2] === GLUE_MAGIC[2] && buf[3] === GLUE_MAGIC[3];
  offset += 4;
  if (!magicValid) {
    throw new Error("Invalid magic number");
  }
  const version = readUint32();
  if (version !== GLUE_VERSION) {
    throw new Error("Invalid version number");
  }
  const name = readString(8);
  const msgProto = GLUE_MESSAGE_PROTOTYPES[name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${name}`);
  }
  const output = { _name: name };
  for (const field of msgProto.fields) {
    const readType = readUint32();
    if (readType === GLUE_DTYPE_NULL) {
      if (!field.isNullable) {
        throw new Error(
          `${name}: Expect field ${field.name} to be non-nullable`
        );
      }
      output[field.name] = null;
      continue;
    }
    if (readType !== TYPE_MAP[field.type]) {
      throw new Error(
        `${name}: Expect field ${field.name} to have type ${field.type}`
      );
    }
    output[field.name] = readField(field);
  }
  return output;
}
function glueSerialize(msg) {
  const msgProto = GLUE_MESSAGE_PROTOTYPES[msg._name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${msg._name}`);
  }
  const bufs = [];
  const writeUint32 = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeInt32 = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setInt32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeFloat = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setFloat32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeBool = (value) => {
    writeUint32(value ? 1 : 0);
  };
  const writeString = (value) => {
    const utf8 = new TextEncoder().encode(value);
    writeUint32(utf8.byteLength);
    bufs.push(utf8);
  };
  const writeRaw = (value) => {
    writeUint32(value.byteLength);
    bufs.push(value);
  };
  const writeArray = (value, writeItem) => {
    writeUint32(value.length);
    for (const item of value) {
      writeItem(item);
    }
  };
  const writeNull = () => {
  };
  bufs.push(GLUE_MAGIC);
  writeUint32(GLUE_VERSION);
  {
    const utf8 = new TextEncoder().encode(msg._name);
    bufs.push(utf8);
  }
  for (const field of msgProto.fields) {
    const val = msg[field.name];
    if (!field.isNullable && (val === null || val === void 0)) {
      throw new Error(
        `${msg._name}: Expect field ${field.name} to be non-nullable`
      );
    }
    if (val === null || val === void 0) {
      writeUint32(GLUE_DTYPE_NULL);
      continue;
    }
    writeUint32(TYPE_MAP[field.type]);
    switch (field.type) {
      case "str":
        writeString(val);
        break;
      case "int":
        writeInt32(val);
        break;
      case "float":
        writeFloat(val);
        break;
      case "bool":
        writeBool(val);
        break;
      case "raw":
        writeRaw(val);
        break;
      case "arr_str":
        writeArray(val, writeString);
        break;
      case "arr_int":
        writeArray(val, writeInt32);
        break;
      case "arr_float":
        writeArray(val, writeFloat);
        break;
      case "arr_bool":
        writeArray(val, writeBool);
        break;
      case "arr_raw":
        writeArray(val, writeRaw);
        break;
      case "null":
        writeNull();
        break;
    }
  }
  const totalLength = bufs.reduce((acc, buf) => acc + buf.byteLength, 0);
  const output = new Uint8Array(totalLength);
  let offset = 0;
  for (const buf of bufs) {
    output.set(buf, offset);
    offset += buf.byteLength;
  }
  return output;
}

// src/wasm/source-map.ts
var WASM_SOURCE_MAP = {
  "default": "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",
  "compat": "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"
};

// src/debug.ts
var cache = /* @__PURE__ */ new Map();
function loadMap(buildKey) {
  return __async(this, null, function* () {
    if (cache.has(buildKey)) return cache.get(buildKey);
    const b64 = WASM_SOURCE_MAP[buildKey];
    if (!b64) throw new Error(`No source map for build "${buildKey}"`);
    const gzipped = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const ds = new DecompressionStream("gzip");
    const writer = ds.writable.getWriter();
    writer.write(gzipped);
    writer.close();
    const buf = yield new Response(ds.readable).arrayBuffer();
    const dv = new DataView(buf);
    const bytes = new Uint8Array(buf);
    const firstId = dv.getUint32(0, true);
    const funcCount = dv.getUint32(4, true);
    const numNames = dv.getUint32(8, true);
    const td = new TextDecoder();
    const names = [];
    let pos = 12;
    for (let i = 0; i < numNames; i++) {
      const len = bytes[pos++];
      names.push(td.decode(bytes.subarray(pos, pos + len)));
      pos += len;
    }
    const funcNames = [];
    for (let i = 0; i < funcCount; i++) {
      const idx = dv.getUint16(pos, true);
      pos += 2;
      funcNames.push(idx === 65535 ? null : names[idx]);
    }
    const entry = { firstId, funcNames };
    cache.set(buildKey, entry);
    return entry;
  });
}
var Debug = {
  /**
   * Resolves a list of wasm function indices to their cleaned symbol names.
   */
  decodeFuncIds: (funcIds, isCompatBuild) => __async(void 0, null, function* () {
    const buildKey = isCompatBuild ? "compat" : "default";
    const { firstId, funcNames } = yield loadMap(buildKey);
    return funcIds.map((funcId) => {
      const i = funcId - firstId;
      const name = i >= 0 && i < funcNames.length && funcNames[i] ? funcNames[i] : "(unknown)";
      return { funcId, name };
    });
  }),
  /**
   * Annotates a wasm stack trace string with resolved function names.
   *
   * Example input from Chrome:
   *   at http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[775]:0x74251
   *   at async blob:http://localhost:8080/53a863cc-7227-45cc-8594-ddbbf5257f20:317:28
   *
   * Example input from Firefox:
   *   @http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[796]:0x7dfe2
   *       at wModuleInit/WebAssembly.promising/< (9b6a2acd-d909-44e2-b021-d42fb9087cfb:15:32) index.js:1433:45
   *
   * Example input from Safari:
   *   2441@wasm-function[2441]
   *       at wrapper (d746f19e-4523-4f36-ba06-d0969acc0b05:22:126009)
   *
   * Example output:
   *   wasm-func[775] (server_response::send)
   */
  decodeStackTrace: (stack, isCompatBuild) => __async(void 0, null, function* () {
    const re = /wasm-function\[(\d+)\]/g;
    const funcIds = [
      ...new Set([...stack.matchAll(re)].map((m) => parseInt(m[1])))
    ];
    if (funcIds.length === 0) return stack;
    const resolved = yield Debug.decodeFuncIds(funcIds, isCompatBuild);
    return resolved.map((r) => {
      if (r.name === "(unknown)") {
        return `    wasm-func[${r.funcId}] (unknown)`;
      }
      return `    wasm-func[${r.funcId}] (${r.name})`;
    }).join("\n");
  })
};

// src/utils.ts
var textDecoder = new TextDecoder();
var URL_PARTS_REGEX = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;
var parseShardNumber = (fnameOrUrl) => {
  const matches = fnameOrUrl.match(URL_PARTS_REGEX);
  if (!matches) {
    return {
      baseURL: fnameOrUrl,
      current: 1,
      total: 1
    };
  } else {
    return {
      baseURL: fnameOrUrl.replace(URL_PARTS_REGEX, ""),
      current: parseInt(matches[1]),
      total: parseInt(matches[2])
    };
  }
};
var sortFileByShard = (blobs) => {
  const isFiles = blobs.every((b) => !!b.name);
  if (isFiles && blobs.length > 1) {
    const files = blobs;
    files.sort((a, b) => {
      const infoA = parseShardNumber(a.name);
      const infoB = parseShardNumber(b.name);
      return infoA.current - infoB.current;
    });
  }
};
var isMmproj = (blob) => __async(void 0, null, function* () {
  const META_NAME = "general.architecture";
  const META_VAL = "clip";
  const tmp = blob.slice(0, 128 * 1024);
  const header = yield tmp.arrayBuffer();
  const buf = new Uint8Array(header);
  const nameBytes = new TextEncoder().encode(META_NAME);
  const valBytes = new TextEncoder().encode(META_VAL);
  let offset = -1;
  outer: for (let i = 0; i <= buf.length - nameBytes.length; i++) {
    for (let j = 0; j < nameBytes.length; j++) {
      if (buf[i + j] !== nameBytes[j]) continue outer;
    }
    offset = i;
    break;
  }
  if (offset === -1) return false;
  if (offset + 8 * 4 + 4 > buf.length) return false;
  const view = new DataView(header);
  const valLen = view.getBigUint64(offset + 8 * 3, true);
  if (valLen !== /* @__PURE__ */ BigInt("4")) return false;
  for (let i = 0; i < valBytes.length; i++) {
    if (buf[offset + 8 * 4 + i] !== valBytes[i]) return false;
  }
  return true;
});
var absoluteUrl = (relativePath) => typeof document === "undefined" ? new URL(relativePath, self.location.href).href : new URL(relativePath, document.baseURI).href;
var padDigits = (number, digits) => {
  return Array(Math.max(digits - String(number).length + 1, 0)).join("0") + number;
};
var sumArr = (arr) => arr.reduce((prev, curr) => prev + curr, 0);
var isString = (value) => !!(value == null ? void 0 : value.startsWith);
var MMPROJ_FILE_NAME = "mmproj.gguf";
var prepareBlobs = (blobsInp) => __async(void 0, null, function* () {
  const blobs = [];
  let blobMmproj = null;
  for (const blob of blobsInp) {
    if (yield isMmproj(blob)) {
      blobMmproj = blob;
    } else {
      blobs.push(blob);
    }
  }
  sortFileByShard(blobs);
  const result = blobs.map((blob, i) => ({
    blob,
    name: `model-${padDigits(i + 1, 5)}-of-${padDigits(blobs.length, 5)}.gguf`
  }));
  if (blobMmproj) {
    result.push({
      blob: blobMmproj,
      name: MMPROJ_FILE_NAME
    });
  }
  return {
    llm: result.filter((f) => f.name !== MMPROJ_FILE_NAME),
    mmproj: blobMmproj ? { blob: blobMmproj, name: MMPROJ_FILE_NAME } : null,
    all: result
  };
});
var isSupportMultiThread = () => ((e) => __async(void 0, null, function* () {
  try {
    return "undefined" != typeof MessageChannel && new MessageChannel().port1.postMessage(new SharedArrayBuffer(1)), WebAssembly.validate(e);
  } catch (e2) {
    return false;
  }
}))(
  new Uint8Array([
    0,
    97,
    115,
    109,
    1,
    0,
    0,
    0,
    1,
    4,
    1,
    96,
    0,
    0,
    3,
    2,
    1,
    0,
    5,
    4,
    1,
    3,
    1,
    1,
    10,
    11,
    1,
    9,
    0,
    65,
    0,
    254,
    16,
    2,
    0,
    26,
    11
  ])
);
var isSupportExceptions = () => __async(void 0, null, function* () {
  return WebAssembly.validate(
    new Uint8Array([
      0,
      97,
      115,
      109,
      1,
      0,
      0,
      0,
      1,
      4,
      1,
      96,
      0,
      0,
      3,
      2,
      1,
      0,
      10,
      8,
      1,
      6,
      0,
      6,
      64,
      25,
      11,
      11
    ])
  );
});
var isSupportSIMD = () => __async(void 0, null, function* () {
  return WebAssembly.validate(
    new Uint8Array([
      0,
      97,
      115,
      109,
      1,
      0,
      0,
      0,
      1,
      5,
      1,
      96,
      0,
      1,
      123,
      3,
      2,
      1,
      0,
      10,
      10,
      1,
      8,
      0,
      65,
      0,
      253,
      15,
      253,
      98,
      11
    ])
  );
});
var isSupportJSPI = () => {
  return !!WebAssembly.Suspending;
};
var isSupportWebGPU = () => {
  return !!navigator.gpu;
};
var isSupportMem64 = () => {
  try {
    new WebAssembly.Memory({
      address: "i64",
      initial: /* @__PURE__ */ BigInt("1")
      // 1 page (64 KiB)
    });
    return true;
  } catch (e) {
    return false;
  }
};
var checkEnvironmentCompatible = () => __async(void 0, null, function* () {
  if (!(yield isSupportExceptions())) {
    throw new Error("WebAssembly runtime does not support exception handling");
  }
  if (!(yield isSupportSIMD())) {
    throw new Error("WebAssembly runtime does not support SIMD");
  }
});
var isFirefox = () => {
  return !!navigator.userAgent.match(/Firefox\/([0-9\.]+)(?:\s|$)/);
};
var GGUF_FILE_REGEX = /^.*\.gguf(?:\?.*)?$/;
var isValidGgufFile = (path) => {
  return GGUF_FILE_REGEX.test(path);
};
var isSafariMobile = () => {
  return !!navigator.userAgent.match(/Version\/([0-9\._]+).*Mobile.*Safari.*/);
};
var createWorker = (workerCode) => {
  const workerURL = URL.createObjectURL(
    isString(workerCode) ? new Blob([workerCode], { type: "text/javascript" }) : workerCode
  );
  return new Worker(workerURL, { type: "module" });
};
var cbToAsyncIter = (fn) => (...args) => {
  let values = [];
  let resolve;
  let reject;
  values.push(
    new Promise((res, rej) => {
      resolve = res;
      reject = rej;
    })
  );
  fn(...args, (val, done, err) => {
    if (err) {
      reject(err);
      return;
    }
    resolve([val, done]);
    values.push(
      new Promise((res, rej) => {
        resolve = res;
        reject = rej;
      })
    );
  });
  return function() {
    return __asyncGenerator(this, null, function* () {
      let val;
      for (let i = 0, done = false; !done; i++) {
        [val, done] = yield new __await(values[i]);
        delete values[i];
        if (val !== void 0) yield val;
      }
    });
  }();
};
var canUseAsyncFileRead = (compat) => isSupportJSPI() || compat;
var needCompat = () => !isSupportJSPI() || !isSupportMem64();

// src/workers-code/generated.ts
var LIBLLAMA_VERSION = "b12081-cb03dae";
var LLAMA_CPP_WORKER_CODE = "// Start the main llama.cpp\nlet wllamaMalloc;\nlet wllamaStart;\nlet wllamaAction;\nlet wllamaExit;\nlet wllamaDebug;\n\nlet Module = null;\nlet isCompat = false;\nlet lastStack = '';\nlet isAborted = false;\nlet hasMultithread = false;\n\n//////////////////////////////////////////////////////////////\n// UTILS\n//////////////////////////////////////////////////////////////\n\n// send message back to main thread\nconst msg = (data, transfer) => postMessage(data, transfer);\n\n// Convert CPP log into JS log\nconst cppLogToJSLog = (line) => {\n  const matched = line.match(/@@(DEBUG|INFO|WARN|ERROR)@@(.*)/);\n  return !!matched\n    ? {\n        level: (matched[1] === 'INFO' ? 'debug' : matched[1]).toLowerCase(),\n        text: matched[2],\n      }\n    : { level: 'log', text: line };\n};\n\nconst getHeapU8 = () => {\n  const buffer = Module.wasmMemory.buffer;\n  return new Uint8Array(buffer);\n};\n\nconst toSizeT = (num) => {\n  return isCompat ? Number(num) : BigInt(num);\n};\n\n// Get module config that forwards stdout/err to main thread\nconst getWModuleConfig = (_argMainScriptBlob) => {\n  var pathConfig = RUN_OPTIONS.pathConfig;\n  var pthreadPoolSize = RUN_OPTIONS.nbThread;\n  var argMainScriptBlob = _argMainScriptBlob;\n\n  isCompat = RUN_OPTIONS.compat;\n  hasMultithread = pthreadPoolSize > 1;\n\n  msg({\n    verb: 'console.debug',\n    args: [\n      `Multithread enabled: ${hasMultithread}, pthreadPoolSize: ${pthreadPoolSize}`,\n    ],\n  });\n\n  if (!pathConfig['wllama.wasm']) {\n    throw new Error('\"wllama.wasm\" is missing in pathConfig');\n  }\n  return {\n    noInitialRun: true,\n    print: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      msg({ verb: 'console.log', args: [text] });\n    },\n    printErr: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      if (text.startsWith('@@STACK@@')) {\n        lastStack = text.slice('@@STACK@@'.length);\n        return;\n      }\n      const logLine = cppLogToJSLog(text);\n      msg({ verb: 'console.' + logLine.level, args: [logLine.text] });\n    },\n    locateFile: function (filename, basePath) {\n      const p = pathConfig[filename];\n      const truncate = (str) =>\n        str.length > 128 ? `${str.substr(0, 128)}...` : str;\n      if (filename.match(/wllama\\.worker\\.js/)) {\n        msg({\n          verb: 'console.error',\n          args: [\n            '\"wllama.worker.js\" is removed from v2.2.1. Hint: make sure to clear browser\\'s cache.',\n          ],\n        });\n      } else {\n        msg({\n          verb: 'console.debug',\n          args: [`Loading \"${filename}\" from \"${truncate(p)}\"`],\n        });\n        return p;\n      }\n    },\n    mainScriptUrlOrBlob: hasMultithread\n      ? argMainScriptBlob\n      : 'throw new Error(\"Multithreading is not enabled\")',\n    pthreadPoolSize: hasMultithread ? pthreadPoolSize : 0,\n    wasmMemory: hasMultithread ? getWasmMemory() : null,\n    onAbort: function (message) {\n      isAborted = true;\n      msg({ verb: 'signal.abort', args: ['abort', message, lastStack, null] });\n    },\n    onExit: function (code) {\n      isAborted = true;\n      const callstack = new Error().stack.toString();\n      msg({\n        verb: 'signal.abort',\n        args: ['abort', 'exit(' + code + ')', callstack, null],\n      });\n    },\n  };\n};\n\n// Get the memory to be used by wasm. (Only used in multi-thread mode)\n// Because we have a weird OOM issue on iOS, we need to try some values\n// See: https://github.com/emscripten-core/emscripten/issues/19144\n//      https://github.com/godotengine/godot/issues/70621\nconst getWasmMemory = () => {\n  let minBytes = 128 * 1024 * 1024;\n  let maxBytes = 4096 * 1024 * 1024;\n  let stepBytes = 128 * 1024 * 1024;\n  while (maxBytes > minBytes) {\n    try {\n      const wasmMemory = new WebAssembly.Memory({\n        initial: toSizeT(minBytes / 65536),\n        maximum: toSizeT(maxBytes / 65536),\n        shared: true,\n        address: isCompat ? undefined : 'i64',\n      });\n      return wasmMemory;\n    } catch (e) {\n      maxBytes -= stepBytes;\n      continue; // retry\n    }\n  }\n  throw new Error('Cannot allocate WebAssembly.Memory');\n};\n\n//////////////////////////////////////////////////////////////\n// HEAPFS PATCH\n//////////////////////////////////////////////////////////////\n\n/**\n * By default, emscripten uses memfs. The way it works is by\n * allocating new Uint8Array in javascript heap. This is not good\n * because it requires files to be copied to wasm heap each time\n * a file is read.\n *\n * HeapFS is an alternative, which resolves this problem by\n * allocating space for file directly inside wasm heap. This\n * allows us to mmap without doing any copy.\n *\n * For llama.cpp, this is great because we use MAP_SHARED\n *\n * Ref: https://github.com/ngxson/wllama/pull/39\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/src/library_memfs.js\n *\n * Note 29/05/2024 @ngxson\n * Due to ftell() being limited to MAX_LONG, we cannot load files bigger than 2^31 bytes (or 2GB)\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/system/lib/libc/musl/src/stdio/ftell.c\n */\n\nconst fsNameToFile = {}; // map Name => File\nconst fsIdToFile = {}; // map ID => File\nlet currFileId = 0;\n\n// Patch and redirect memfs calls to wllama\nconst patchHeapFS = () => {\n  const m = Module;\n  // save functions\n  m.MEMFS.stream_ops._read = m.MEMFS.stream_ops.read;\n  m.MEMFS.stream_ops._write = m.MEMFS.stream_ops.write;\n  m.MEMFS.stream_ops._llseek = m.MEMFS.stream_ops.llseek;\n  m.MEMFS.stream_ops._allocate = m.MEMFS.stream_ops.allocate;\n  m.MEMFS.stream_ops._mmap = m.MEMFS.stream_ops.mmap;\n  m.MEMFS.stream_ops._msync = m.MEMFS.stream_ops.msync;\n\n  const patchStream = (stream) => {\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const ptr = Number(f.ptr);\n      stream.node.contents = getHeapU8().subarray(ptr, ptr + f.size);\n      stream.node.usedBytes = f.size;\n    }\n  };\n\n  // replace \"read\" functions\n  m.MEMFS.stream_ops.read = function (\n    stream,\n    buffer,\n    offset,\n    length,\n    position\n  ) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._read(stream, buffer, offset, length, position);\n  };\n  m.MEMFS.ops_table.file.stream.read = m.MEMFS.stream_ops.read;\n\n  // replace \"llseek\" functions\n  m.MEMFS.stream_ops.llseek = function (stream, offset, whence) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._llseek(stream, offset, whence);\n  };\n  m.MEMFS.ops_table.file.stream.llseek = m.MEMFS.stream_ops.llseek;\n\n  // replace \"mmap\" functions\n  m.MEMFS.stream_ops.mmap = function (stream, length, position, prot, flags) {\n    patchStream(stream);\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const mmapPtr = f.ptr + toSizeT(position);\n      return {\n        ptr: mmapPtr,\n        allocated: false,\n      };\n    } else {\n      return m.MEMFS.stream_ops._mmap(stream, length, position, prot, flags);\n    }\n  };\n  m.MEMFS.ops_table.file.stream.mmap = m.MEMFS.stream_ops.mmap;\n\n  // mount FS\n  m.FS.mkdir('/models');\n  m.FS.mount(m.MEMFS, { root: '.' }, '/models');\n  // slot save files (action \"slots\"), moved to and from the main thread as bytes\n  m.FS.mkdir('/slots');\n};\n\n// Allocate a new file in wllama heapfs, returns file ID\nconst heapfsAlloc = (name, size, allocBuffer) => {\n  if (size < 1) {\n    throw new Error('File size must be bigger than 0');\n  }\n  const m = Module;\n  const ptr = toSizeT(allocBuffer ? m.mmapAlloc(size) : 0);\n  const file = {\n    ptr: ptr,\n    size: size,\n    id: currFileId++,\n  };\n  fsIdToFile[file.id] = file;\n  fsNameToFile[name] = file;\n  return file.id;\n};\n\n// Add new file to wllama heapfs, return number of written bytes\nconst heapfsWrite = (id, buffer, offset) => {\n  if (fsIdToFile[id]) {\n    const { ptr, size } = fsIdToFile[id];\n    const afterWriteByte = offset + buffer.byteLength;\n    if (afterWriteByte > size) {\n      throw new Error(\n        `File ID ${id} write out of bound, afterWriteByte = ${afterWriteByte} while size = ${size}`\n      );\n    }\n    getHeapU8().set(buffer, Number(ptr) + offset);\n    return buffer.byteLength;\n  } else {\n    throw new Error(`File ID ${id} not found in heapfs`);\n  }\n};\n\n//////////////////////////////////////////////////////////////\n// ASYNC FILE READ\n//////////////////////////////////////////////////////////////\n\nlet isAwaitReading = false;\nlet pendingReadPromise = null;\nlet pendingReadResolve = null;\nlet pendingReadReject = null;\n\nconst _stripModelsPrefix = (path) => path.replace(/^\\/?models\\//, '');\n\n// Called from EM_ASYNC_JS stub in wllama-fs.h (path is already a JS string)\nconst _wllama_js_file_read = async (path, offset, req_size, out_ptr) => {\n  const name = _stripModelsPrefix(path);\n\n  pendingReadPromise = new Promise((res, rej) => {\n    pendingReadResolve = res;\n    pendingReadReject = rej;\n  });\n  isAwaitReading = true;\n\n  postMessage({ verb: 'fs.read_req', args: [name, offset, req_size] });\n\n  let data;\n  try {\n    data = await pendingReadPromise;\n  } finally {\n    isAwaitReading = false;\n    pendingReadResolve = null;\n    pendingReadReject = null;\n  }\n\n  const bytes = new Uint8Array(data);\n  getHeapU8().set(bytes, out_ptr);\n  return toSizeT(bytes.length);\n};\n\n//////////////////////////////////////////////////////////////\n// MAIN CODE\n//////////////////////////////////////////////////////////////\n\nconst callWrapper = (name, ret, args, isAsync) => {\n  const fn = Module.cwrap(\n    name,\n    ret,\n    args,\n    isAsync ? { async: true } : undefined\n  );\n  return async (action, req) => {\n    // console.log(`Calling ${name} with action:`, action, 'and req:', req);\n    let result;\n    try {\n      if (args.length === 2) {\n        result = isAsync ? await fn(action, req) : fn(action, req);\n      } else {\n        result = fn();\n      }\n    } catch (ex) {\n      console.error(ex);\n      throw ex;\n    }\n    return result;\n  };\n};\n\n// re-entering the wasm while a call is suspended (JSPI / asyncify) corrupts its state, so only one call runs at a time and the rest wait in the queue\nlet wasmCallBusy = false;\nconst wasmCallQueue = [];\n\nconst runWasmCall = async (callbackId, fn) => {\n  if (isAborted) {\n    // the wasm is dead, fail fast instead of calling into it\n    msg({ callbackId, err: 'wllama has crashed, please reload the module' });\n    return;\n  }\n  if (wasmCallBusy) {\n    wasmCallQueue.push({ callbackId, fn });\n    return;\n  }\n  wasmCallBusy = true;\n  try {\n    await fn();\n  } finally {\n    wasmCallBusy = false;\n    if (isAborted) {\n      // do not touch the wasm again after it aborted; the main thread already rejected the queued tasks\n      wasmCallQueue.length = 0;\n    } else {\n      const next = wasmCallQueue.shift();\n      if (next) runWasmCall(next.callbackId, next.fn);\n    }\n  }\n};\n\nconst runAction = async (data) => {\n  const { args, callbackId } = data;\n  const argAction = args[0];\n  const argEncodedMsg = args[1];\n  try {\n    const inputPtr = await wllamaMalloc(toSizeT(argEncodedMsg.byteLength), 0);\n    // copy data to wasm heap\n    const inputBuffer = new Uint8Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      argEncodedMsg.byteLength\n    );\n    inputBuffer.set(argEncodedMsg, 0);\n    const outputPtr = await wllamaAction(argAction, inputPtr);\n    // length of output buffer is written at the first 4 bytes of input buffer\n    const outputLen = new Uint32Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      1\n    )[0];\n    // copy the output buffer to JS heap\n    const outputBuffer = new Uint8Array(outputLen);\n    const outputSrcView = new Uint8Array(\n      getHeapU8().buffer,\n      Number(outputPtr),\n      outputLen\n    );\n    outputBuffer.set(outputSrcView, 0); // copy it\n    msg({ callbackId, result: outputBuffer }, [outputBuffer.buffer]);\n  } catch (err) {\n    handleError(err);\n  }\n};\n\nfunction handleError(err) {\n  // If WASM already aborted, onAbort already sent signal.abort; skip to avoid\n  // re-reporting the resulting WebAssembly.RuntimeError as a JS exception.\n  if (isAborted) return;\n\n  const message = err ? err.message || String(err) : 'Unknown error';\n  const stack = err ? err.stack || String(err) : '';\n  msg({\n    verb: 'signal.abort',\n    args: ['exception', message, stack, err],\n  });\n}\n\nonmessage = async (e) => {\n  if (!e.data) return;\n  const { verb, args, callbackId } = e.data;\n\n  // fs.read_res arrives while wasm is JSPI-suspended; resolve the pending promise.\n  if (verb === 'fs.read_res') {\n    if (pendingReadResolve) {\n      pendingReadResolve(args[0]);\n    }\n    return;\n  }\n\n  // Guard: while awaiting a file read, reject any other incoming task.\n  if (isAwaitReading) {\n    if (callbackId) {\n      msg({\n        callbackId,\n        err: 'Worker is suspended waiting for file data (JSPI)',\n      });\n    }\n    return;\n  }\n\n  if (!callbackId) {\n    msg({ verb: 'console.error', args: ['callbackId is required', e.data] });\n    return;\n  }\n\n  if (verb === 'module.init') {\n    const argMainScriptBlob = args[0];\n    const argUseAsyncFile = args[1];\n    try {\n      Module = getWModuleConfig(argMainScriptBlob);\n      Module.preRun = () => {\n        if (argUseAsyncFile) {\n          Module.ENV['USE_ASYNC_FILE'] = '1';\n        }\n      };\n      Module.onRuntimeInitialized = () => {\n        // async call once module is ready\n        // init FS\n        patchHeapFS();\n        // init cwrap\n        const pointer = isCompat ? 'number' : 'bigint';\n        // TODO: note sure why emscripten cannot bind if there is only 1 argument\n        wllamaMalloc = callWrapper('wllama_malloc', pointer, [\n          'number',\n          pointer,\n        ]);\n        wllamaStart = callWrapper('wllama_start', 'string', [], true);\n        wllamaAction = callWrapper(\n          'wllama_action',\n          pointer,\n          ['string', pointer],\n          true\n        );\n        wllamaExit = callWrapper('wllama_exit', 'string', []);\n        wllamaDebug = callWrapper('wllama_debug', 'string', []);\n        msg({ callbackId, result: null });\n      };\n      wModuleInit();\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.alloc') {\n    const argFilename = args[0];\n    const argSize = args[1];\n    const argAllocBuffer = args[2];\n    try {\n      // create blank file\n      const emptyBuffer = new ArrayBuffer(0);\n      Module['FS_createDataFile'](\n        '/models',\n        argFilename,\n        emptyBuffer,\n        true,\n        true,\n        true\n      );\n      // alloc data on heap\n      const fileId = heapfsAlloc(argFilename, argSize, argAllocBuffer);\n      msg({ callbackId, result: { fileId } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.write') {\n    const argFileId = args[0];\n    const argBuffer = args[1];\n    const argOffset = args[2];\n    try {\n      const writtenBytes = heapfsWrite(argFileId, argBuffer, argOffset);\n      msg({ callbackId, result: { writtenBytes } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_read') {\n    const argFilename = args[0];\n    try {\n      const path = '/slots/' + argFilename;\n      const data = Module.FS.readFile(path);\n      Module.FS.unlink(path);\n      msg({ callbackId, result: data }, [data.buffer]);\n    } catch (err) {\n      msg({ callbackId, err: `cannot read slot file ${argFilename}: ${err}` });\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_write') {\n    const argFilename = args[0];\n    const argData = args[1];\n    try {\n      Module.FS.writeFile('/slots/' + argFilename, argData);\n      msg({ callbackId, result: null });\n    } catch (err) {\n      msg({ callbackId, err: `cannot write slot file ${argFilename}: ${err}` });\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_remove') {\n    try {\n      Module.FS.unlink('/slots/' + args[0]);\n    } catch (err) {\n      // already removed\n    }\n    msg({ callbackId, result: null });\n    return;\n  }\n\n  if (verb === 'wllama.start') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaStart();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.action') {\n    await runWasmCall(callbackId, () => runAction(e.data));\n    return;\n  }\n\n  if (verb === 'wllama.exit') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaExit();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.debug') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaDebug();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n};\n";
var OPFS_UTILS_WORKER_CODE = "let accessHandle;\nlet abortController = new AbortController();\n\nasync function openFile(filename) {\n  const opfsRoot = await navigator.storage.getDirectory();\n  const cacheDir = await opfsRoot.getDirectoryHandle('cache', { create: true });\n  const fileHandler = await cacheDir.getFileHandle(filename, { create: true });\n  accessHandle = await fileHandler.createSyncAccessHandle();\n  accessHandle.truncate(0); // clear file content\n}\n\nasync function writeFile(buf) {\n  accessHandle.write(buf);\n}\n\nasync function closeFile() {\n  accessHandle.flush();\n  accessHandle.close();\n}\n\nasync function writeTextFile(filename, str) {\n  await openFile(filename);\n  await writeFile(new TextEncoder().encode(str));\n  await closeFile();\n}\n\nconst throttled = (func, delay) => {\n  let lastRun = 0;\n  return (...args) => {\n    const now = Date.now();\n    if (now - lastRun > delay) {\n      lastRun = now;\n      func.apply(null, args);\n    }\n  };\n};\n\nconst assertNonNull = (val) => {\n  if (val === null || val === undefined) {\n    throw new Error('OPFS Worker: Assertion failed');\n  }\n};\n\n// respond to main thread\nconst resOK = () => postMessage({ ok: true });\nconst resProgress = (loaded, total) =>\n  postMessage({ progress: { loaded, total } });\nconst resErr = (err) => postMessage({ err });\n\nonmessage = async (e) => {\n  try {\n    if (!e.data) return;\n\n    /**\n     * @param {Object} e.data\n     *\n     * Fine-control FS actions:\n     * - { action: 'open', filename: 'string' }\n     * - { action: 'write', buf: ArrayBuffer }\n     * - { action: 'close' }\n     *\n     * Simple write API:\n     * - { action: 'write-simple', filename: 'string', buf: ArrayBuffer }\n     *\n     * Download API:\n     * - { action: 'download', url: 'string', filename: 'string', options: Object, metadataFileName: 'string' }\n     * - { action: 'download-abort' }\n     */\n    const {\n      action,\n      filename,\n      buf,\n      url,\n      options,\n      metadataFileName,\n      metadataAdditional,\n    } = e.data;\n\n    if (action === 'open') {\n      assertNonNull(filename);\n      await openFile(filename);\n      return resOK();\n    } else if (action === 'write') {\n      assertNonNull(buf);\n      await writeFile(buf);\n      return resOK();\n    } else if (action === 'close') {\n      await closeFile();\n      return resOK();\n    } else if (action === 'write-simple') {\n      assertNonNull(filename);\n      assertNonNull(buf);\n      await openFile(filename);\n      await writeFile(buf);\n      await closeFile();\n      return resOK();\n    } else if (action === 'download') {\n      assertNonNull(url);\n      assertNonNull(filename);\n      assertNonNull(metadataFileName);\n      assertNonNull(options);\n      assertNonNull(options.aborted);\n      abortController = new AbortController();\n      if (options.aborted) abortController.abort();\n      const response = await fetch(url, {\n        ...options,\n        signal: abortController.signal,\n      });\n      const contentLength = response.headers.get('content-length');\n      const etag = (response.headers.get('etag') || '').replace(\n        /[^A-Za-z0-9]/g,\n        ''\n      );\n      const total = parseInt(contentLength, 10);\n      const reader = response.body.getReader();\n      await openFile(filename);\n      let loaded = 0;\n      const throttledProgress = throttled(resProgress, 100);\n      while (true) {\n        const { done, value } = await reader.read();\n        if (done) break;\n        loaded += value.byteLength;\n        await writeFile(value);\n        throttledProgress(loaded, total);\n      }\n      resProgress(total, total); // 100% done\n      await closeFile();\n      // make sure this is in-sync with CacheEntryMetadata\n      await writeTextFile(\n        metadataFileName,\n        JSON.stringify({\n          originalURL: url,\n          originalSize: total,\n          etag,\n          ...metadataAdditional,\n        })\n      );\n      return resOK();\n    } else if (action === 'download-abort') {\n      if (abortController) {\n        abortController.abort();\n      }\n      return;\n    }\n\n    throw new Error('OPFS Worker: Invalid action', e.data);\n  } catch (err) {\n    return resErr(err);\n  }\n};\n";
var WLLAMA_EMSCRIPTEN_CODE = 'var Module=typeof Module!="undefined"?Module:{};var ENVIRONMENT_IS_WEB=!!globalThis.window;var ENVIRONMENT_IS_WORKER=!!globalThis.WorkerGlobalScope;var ENVIRONMENT_IS_NODE=globalThis.process?.versions?.node&&globalThis.process?.type!="renderer";var ENVIRONMENT_IS_PTHREAD=ENVIRONMENT_IS_WORKER&&self.name?.startsWith("em-pthread");if(ENVIRONMENT_IS_NODE){var worker_threads=require("worker_threads");global.Worker=worker_threads.Worker;ENVIRONMENT_IS_WORKER=!worker_threads.isMainThread;ENVIRONMENT_IS_PTHREAD=ENVIRONMENT_IS_WORKER&&worker_threads["workerData"]=="em-pthread"}var arguments_=[];var thisProgram="./this.program";var quit_=(status,toThrow)=>{throw toThrow};var _scriptName=globalThis.document?.currentScript?.src;if(typeof __filename!="undefined"){_scriptName=__filename}else if(ENVIRONMENT_IS_WORKER){_scriptName=self.location.href}var scriptDirectory="";function locateFile(path){if(Module["locateFile"]){return Module["locateFile"](path,scriptDirectory)}return scriptDirectory+path}var readAsync,readBinary;if(ENVIRONMENT_IS_NODE){var fs=require("fs");scriptDirectory=__dirname+"/";readBinary=filename=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename);return ret};readAsync=async(filename,binary=true)=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename,binary?undefined:"utf8");return ret};if(process.argv.length>1){thisProgram=process.argv[1].replace(/\\\\/g,"/")}arguments_=process.argv.slice(2);if(typeof module!="undefined"){module["exports"]=Module}quit_=(status,toThrow)=>{process.exitCode=status;throw toThrow}}else if(ENVIRONMENT_IS_WEB||ENVIRONMENT_IS_WORKER){try{scriptDirectory=new URL(".",_scriptName).href}catch{}if(!ENVIRONMENT_IS_NODE){if(ENVIRONMENT_IS_WORKER){readBinary=url=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,false);xhr.responseType="arraybuffer";xhr.send(null);return new Uint8Array(xhr.response)}}readAsync=async url=>{if(isFileURI(url)){return new Promise((resolve,reject)=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,true);xhr.responseType="arraybuffer";xhr.onload=()=>{if(xhr.status==200||xhr.status==0&&xhr.response){resolve(xhr.response);return}reject(xhr.status)};xhr.onerror=reject;xhr.send(null)})}var response=await fetch(url,{credentials:"same-origin"});if(response.ok){return response.arrayBuffer()}throw new Error(response.status+" : "+response.url)}}}else{}var defaultPrint=console.log.bind(console);var defaultPrintErr=console.error.bind(console);if(ENVIRONMENT_IS_NODE){var utils=require("util");var stringify=a=>typeof a=="object"?utils.inspect(a):a;defaultPrint=(...args)=>fs.writeSync(1,args.map(stringify).join(" ")+"\\n");defaultPrintErr=(...args)=>fs.writeSync(2,args.map(stringify).join(" ")+"\\n")}var out=defaultPrint;var err=defaultPrintErr;var wasmBinary;var wasmModule;var ABORT=false;var EXITSTATUS;function assert(condition,text){if(!condition){abort(text)}}var isFileURI=filename=>filename.startsWith("file://");function growMemViews(){if(wasmMemory.buffer!=HEAP8.buffer){updateMemoryViews()}}if(ENVIRONMENT_IS_NODE&&ENVIRONMENT_IS_PTHREAD){var parentPort=worker_threads["parentPort"];parentPort.on("message",msg=>global.onmessage?.({data:msg}));Object.assign(globalThis,{self:global,postMessage:msg=>parentPort["postMessage"](msg)});process.on("uncaughtException",err=>{postMessage({cmd:"uncaughtException",error:err});process.exit(1)})}var startWorker;if(ENVIRONMENT_IS_PTHREAD){var initializedJS=false;self.onunhandledrejection=e=>{throw e.reason||e};async function handleMessage(e){try{var msgData=e["data"];var cmd=msgData.cmd;if(cmd==="load"){let messageQueue=[];self.onmessage=e=>messageQueue.push(e);startWorker=()=>{postMessage({cmd:"loaded"});for(let msg of messageQueue){handleMessage(msg)}self.onmessage=handleMessage};for(const handler of msgData.handlers){if(!Module[handler]||Module[handler].proxy){Module[handler]=(...args)=>{postMessage({cmd:"callHandler",handler,args})};if(handler=="print")out=Module[handler];if(handler=="printErr")err=Module[handler]}}wasmMemory=msgData.wasmMemory;updateMemoryViews();wasmModule=msgData.wasmModule;createWasm();run()}else if(cmd==="run"){establishStackSpace(msgData.pthread_ptr);__emscripten_thread_init(msgData.pthread_ptr,0,0,1,0,0);PThread.threadInitTLS();__emscripten_thread_mailbox_await(msgData.pthread_ptr);if(!initializedJS){initializedJS=true}try{await invokeEntryPoint(msgData.start_routine,msgData.arg)}catch(ex){if(ex!="unwind"){throw ex}}}else if(msgData.target==="setimmediate"){}else if(cmd==="checkMailbox"){if(initializedJS){checkMailbox()}}else if(cmd){err(`worker: received unknown command ${cmd}`);err(msgData)}}catch(ex){__emscripten_thread_crashed();throw ex}}self.onmessage=handleMessage}var HEAP8,HEAPU8,HEAP16,HEAPU16,HEAP32,HEAPU32,HEAPF32,HEAPF64;var HEAP64,HEAPU64;var runtimeInitialized=false;function updateMemoryViews(){var b=wasmMemory.buffer;HEAP8=new Int8Array(b);HEAP16=new Int16Array(b);Module["HEAPU8"]=HEAPU8=new Uint8Array(b);HEAPU16=new Uint16Array(b);HEAP32=new Int32Array(b);HEAPU32=new Uint32Array(b);HEAPF32=new Float32Array(b);HEAPF64=new Float64Array(b);HEAP64=new BigInt64Array(b);HEAPU64=new BigUint64Array(b)}function initMemory(){if(ENVIRONMENT_IS_PTHREAD){return}if(Module["wasmMemory"]){wasmMemory=Module["wasmMemory"]}else{var INITIAL_MEMORY=Module["INITIAL_MEMORY"]||134217728;wasmMemory=new WebAssembly.Memory({initial:BigInt(INITIAL_MEMORY/65536),maximum:65536n,shared:true,address:"i64"})}updateMemoryViews()}function preRun(){if(Module["preRun"]){if(typeof Module["preRun"]=="function")Module["preRun"]=[Module["preRun"]];while(Module["preRun"].length){addOnPreRun(Module["preRun"].shift())}}callRuntimeCallbacks(onPreRuns)}function initRuntime(){runtimeInitialized=true;if(ENVIRONMENT_IS_PTHREAD)return startWorker();if(!Module["noFSInit"]&&!FS.initialized)FS.init();TTY.init();wasmExports["__wasm_call_ctors"]();FS.ignorePermissions=false}function preMain(){}function postRun(){if(ENVIRONMENT_IS_PTHREAD){return}if(Module["postRun"]){if(typeof Module["postRun"]=="function")Module["postRun"]=[Module["postRun"]];while(Module["postRun"].length){addOnPostRun(Module["postRun"].shift())}}callRuntimeCallbacks(onPostRuns)}function abort(what){Module["onAbort"]?.(what);what="Aborted("+what+")";err(what);ABORT=true;what+=". Build with -sASSERTIONS for more info.";if(runtimeInitialized){___trap()}var e=new WebAssembly.RuntimeError(what);throw e}var wasmBinaryFile;function findWasmBinary(){return locateFile("wllama.wasm")}function getBinarySync(file){if(file==wasmBinaryFile&&wasmBinary){return new Uint8Array(wasmBinary)}if(readBinary){return readBinary(file)}throw"both async and sync fetching of the wasm failed"}async function getWasmBinary(binaryFile){if(!wasmBinary){try{var response=await readAsync(binaryFile);return new Uint8Array(response)}catch{}}return getBinarySync(binaryFile)}async function instantiateArrayBuffer(binaryFile,imports){try{var binary=await getWasmBinary(binaryFile);var instance=await WebAssembly.instantiate(binary,imports);return instance}catch(reason){err(`failed to asynchronously prepare wasm: ${reason}`);abort(reason)}}async function instantiateAsync(binary,binaryFile,imports){if(!binary&&!isFileURI(binaryFile)&&!ENVIRONMENT_IS_NODE){try{var response=fetch(binaryFile,{credentials:"same-origin"});var instantiationResult=await WebAssembly.instantiateStreaming(response,imports);return instantiationResult}catch(reason){err(`wasm streaming compile failed: ${reason}`);err("falling back to ArrayBuffer instantiation")}}return instantiateArrayBuffer(binaryFile,imports)}function getWasmImports(){assignWasmImports();if(!wasmImports.__instrumented){wasmImports.__instrumented=true;Asyncify.instrumentWasmImports(wasmImports)}var imports={env:wasmImports,wasi_snapshot_preview1:wasmImports};return imports}async function createWasm(){function receiveInstance(instance,module){wasmExports=instance.exports;wasmExports=Asyncify.instrumentWasmExports(wasmExports);wasmExports=applySignatureConversions(wasmExports);registerTLSInit(wasmExports["_emscripten_tls_init"]);assignWasmExports(wasmExports);wasmModule=module;removeRunDependency("wasm-instantiate");return wasmExports}addRunDependency("wasm-instantiate");function receiveInstantiationResult(result){return receiveInstance(result["instance"],result["module"])}var info=getWasmImports();if(Module["instantiateWasm"]){return new Promise((resolve,reject)=>{Module["instantiateWasm"](info,(inst,mod)=>{resolve(receiveInstance(inst,mod))})})}if(ENVIRONMENT_IS_PTHREAD){var instance=new WebAssembly.Instance(wasmModule,getWasmImports());return receiveInstance(instance,wasmModule)}wasmBinaryFile??=findWasmBinary();var result=await instantiateAsync(wasmBinary,wasmBinaryFile,info);var exports=receiveInstantiationResult(result);return exports}class ExitStatus{name="ExitStatus";constructor(status){this.message=`Program terminated with exit(${status})`;this.status=status}}var terminateWorker=worker=>{worker.terminate();worker.onmessage=e=>{}};var cleanupThread=pthread_ptr=>{var worker=PThread.pthreads[pthread_ptr];PThread.returnWorkerToPool(worker)};var callRuntimeCallbacks=callbacks=>{while(callbacks.length>0){callbacks.shift()(Module)}};var onPreRuns=[];var addOnPreRun=cb=>onPreRuns.push(cb);var runDependencies=0;var dependenciesFulfilled=null;var removeRunDependency=id=>{runDependencies--;Module["monitorRunDependencies"]?.(runDependencies);if(runDependencies==0){if(dependenciesFulfilled){var callback=dependenciesFulfilled;dependenciesFulfilled=null;callback()}}};var addRunDependency=id=>{runDependencies++;Module["monitorRunDependencies"]?.(runDependencies)};var spawnThread=threadParams=>{var worker=PThread.getNewWorker();if(!worker){return 6}PThread.runningWorkers.push(worker);PThread.pthreads[threadParams.pthread_ptr]=worker;worker.pthread_ptr=threadParams.pthread_ptr;var msg={cmd:"run",start_routine:threadParams.startRoutine,arg:threadParams.arg,pthread_ptr:threadParams.pthread_ptr};if(ENVIRONMENT_IS_NODE){worker.unref()}worker.postMessage(msg,threadParams.transferList);return 0};var runtimeKeepaliveCounter=0;var keepRuntimeAlive=()=>noExitRuntime||runtimeKeepaliveCounter>0;var stackSave=()=>_emscripten_stack_get_current();var stackRestore=val=>__emscripten_stack_restore(val);var stackAlloc=sz=>__emscripten_stack_alloc(sz);var proxyToMainThread=(funcIndex,emAsmAddr,sync,...callArgs)=>{var serializedNumCallArgs=callArgs.length*2;var sp=stackSave();var args=stackAlloc(serializedNumCallArgs*8);var b=args/8;for(var i=0;i<callArgs.length;i++){var arg=callArgs[i];if(typeof arg=="bigint"){(growMemViews(),HEAP64)[b+2*i]=1n;(growMemViews(),HEAP64)[b+2*i+1]=arg}else{(growMemViews(),HEAP64)[b+2*i]=0n;(growMemViews(),HEAPF64)[b+2*i+1]=arg}}var rtn=__emscripten_run_js_on_main_thread(funcIndex,emAsmAddr,serializedNumCallArgs,args,sync);stackRestore(sp);return rtn};function _proc_exit(code){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(0,0,1,code);EXITSTATUS=code;if(!keepRuntimeAlive()){PThread.terminateAllThreads();Module["onExit"]?.(code);ABORT=true}quit_(code,new ExitStatus(code))}function exitOnMainThread(returnCode){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(1,0,0,returnCode);_exit(returnCode)}var exitJS=(status,implicit)=>{EXITSTATUS=status;if(ENVIRONMENT_IS_PTHREAD){exitOnMainThread(status);throw"unwind"}_proc_exit(status)};var _exit=exitJS;var PThread={unusedWorkers:[],runningWorkers:[],tlsInitFunctions:[],pthreads:{},init(){if(!ENVIRONMENT_IS_PTHREAD){PThread.initMainThread()}},initMainThread(){var pthreadPoolSize=Module["pthreadPoolSize"];while(pthreadPoolSize--){PThread.allocateUnusedWorker()}addOnPreRun(async()=>{var pthreadPoolReady=PThread.loadWasmModuleToAllWorkers();addRunDependency("loading-workers");await pthreadPoolReady;removeRunDependency("loading-workers")})},terminateAllThreads:()=>{for(var worker of PThread.runningWorkers){terminateWorker(worker)}for(var worker of PThread.unusedWorkers){terminateWorker(worker)}PThread.unusedWorkers=[];PThread.runningWorkers=[];PThread.pthreads={}},returnWorkerToPool:worker=>{var pthread_ptr=worker.pthread_ptr;delete PThread.pthreads[pthread_ptr];PThread.unusedWorkers.push(worker);PThread.runningWorkers.splice(PThread.runningWorkers.indexOf(worker),1);worker.pthread_ptr=0;__emscripten_thread_free_data(pthread_ptr)},threadInitTLS(){PThread.tlsInitFunctions.forEach(f=>f())},loadWasmModuleToWorker:worker=>new Promise(onFinishedLoading=>{worker.onmessage=e=>{var d=e["data"];var cmd=d.cmd;if(d.targetThread&&d.targetThread!=_pthread_self()){var targetWorker=PThread.pthreads[d.targetThread];if(targetWorker){targetWorker.postMessage(d,d.transferList)}else{err(`Internal error! Worker sent a message "${cmd}" to target pthread ${d.targetThread}, but that thread no longer exists!`)}return}if(cmd==="checkMailbox"){checkMailbox()}else if(cmd==="spawnThread"){spawnThread(d)}else if(cmd==="cleanupThread"){callUserCallback(()=>cleanupThread(d.thread))}else if(cmd==="loaded"){worker.loaded=true;if(ENVIRONMENT_IS_NODE&&!worker.pthread_ptr){worker.unref()}onFinishedLoading(worker)}else if(d.target==="setimmediate"){worker.postMessage(d)}else if(cmd==="uncaughtException"){worker.onerror(d.error)}else if(cmd==="callHandler"){Module[d.handler](...d.args)}else if(cmd){err(`worker sent an unknown command ${cmd}`)}};worker.onerror=e=>{var message="worker sent an error!";err(`${message} ${e.filename}:${e.lineno}: ${e.message}`);throw e};if(ENVIRONMENT_IS_NODE){worker.on("message",data=>worker.onmessage({data}));worker.on("error",e=>worker.onerror(e))}var handlers=[];var knownHandlers=["onExit","onAbort","print","printErr"];for(var handler of knownHandlers){if(Module.propertyIsEnumerable(handler)){handlers.push(handler)}}worker.postMessage({cmd:"load",handlers,wasmMemory,wasmModule})}),async loadWasmModuleToAllWorkers(){if(ENVIRONMENT_IS_PTHREAD){return}let pthreadPoolReady=Promise.all(PThread.unusedWorkers.map(PThread.loadWasmModuleToWorker));return pthreadPoolReady},allocateUnusedWorker(){var worker;var pthreadMainJs=_scriptName;if(Module["mainScriptUrlOrBlob"]){pthreadMainJs=Module["mainScriptUrlOrBlob"];if(typeof pthreadMainJs!="string"){pthreadMainJs=URL.createObjectURL(pthreadMainJs)}}worker=new Worker(pthreadMainJs,{workerData:"em-pthread",name:"em-pthread"});PThread.unusedWorkers.push(worker)},getNewWorker(){if(PThread.unusedWorkers.length==0){PThread.allocateUnusedWorker();PThread.loadWasmModuleToWorker(PThread.unusedWorkers[0])}return PThread.unusedWorkers.pop()}};var onPostRuns=[];var addOnPostRun=cb=>onPostRuns.push(cb);function establishStackSpace(pthread_ptr){var stackHigh=Number((growMemViews(),HEAPU64)[(pthread_ptr+88)/8]);var stackSize=Number((growMemViews(),HEAPU64)[(pthread_ptr+96)/8]);var stackLow=stackHigh-stackSize;_emscripten_stack_set_limits(stackHigh,stackLow);stackRestore(stackHigh)}var wasmTableMirror=[];var getWasmTableEntry=funcPtr=>{funcPtr=Number(funcPtr);var func=wasmTableMirror[funcPtr];if(!func){wasmTableMirror[funcPtr]=func=wasmTable.get(BigInt(funcPtr));if(Asyncify.isAsyncExport(func)){wasmTableMirror[funcPtr]=func=Asyncify.makeAsyncFunction(func)}}return func};var invokeEntryPoint=async(ptr,arg)=>{runtimeKeepaliveCounter=0;noExitRuntime=0;var result=(a1=>WebAssembly.promising(getWasmTableEntry(ptr)).call(null,BigInt(a1)))(arg);function finish(result){if(keepRuntimeAlive()){EXITSTATUS=result;return}__emscripten_thread_exit(result)}result=await result;finish(result)};invokeEntryPoint.isAsync=true;var noExitRuntime=true;var registerTLSInit=tlsInitFunc=>PThread.tlsInitFunctions.push(tlsInitFunc);var wasmMemory;function pthreadCreateProxied(pthread_ptr,attr,startRoutine,arg){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(2,0,1,pthread_ptr,attr,startRoutine,arg);return ___pthread_create_js(pthread_ptr,attr,startRoutine,arg)}var _emscripten_has_threading_support=()=>!!globalThis.SharedArrayBuffer;var INT53_MAX=9007199254740992;var INT53_MIN=-9007199254740992;var bigintToI53Checked=num=>num<INT53_MIN||num>INT53_MAX?NaN:Number(num);function ___pthread_create_js(pthread_ptr,attr,startRoutine,arg){pthread_ptr=bigintToI53Checked(pthread_ptr);attr=bigintToI53Checked(attr);startRoutine=bigintToI53Checked(startRoutine);arg=bigintToI53Checked(arg);if(!_emscripten_has_threading_support()){return 6}var transferList=[];var error=0;if(ENVIRONMENT_IS_PTHREAD&&(transferList.length===0||error)){return pthreadCreateProxied(pthread_ptr,attr,startRoutine,arg)}if(error)return error;var threadParams={startRoutine,pthread_ptr,arg,transferList};if(ENVIRONMENT_IS_PTHREAD){threadParams.cmd="spawnThread";postMessage(threadParams,transferList);return 0}return spawnThread(threadParams)}var syscallGetVarargP=()=>{var ret=Number((growMemViews(),HEAPU64)[SYSCALLS.varargs/8]);SYSCALLS.varargs+=8;return ret};var syscallGetVarargI=()=>{var ret=(growMemViews(),HEAP32)[+SYSCALLS.varargs/4];SYSCALLS.varargs+=4;return ret};var PATH={isAbs:path=>path.charAt(0)==="/",splitPath:filename=>{var splitPathRe=/^(\\/?|)([\\s\\S]*?)((?:\\.{1,2}|[^\\/]+?|)(\\.[^.\\/]*|))(?:[\\/]*)$/;return splitPathRe.exec(filename).slice(1)},normalizeArray:(parts,allowAboveRoot)=>{var up=0;for(var i=parts.length-1;i>=0;i--){var last=parts[i];if(last==="."){parts.splice(i,1)}else if(last===".."){parts.splice(i,1);up++}else if(up){parts.splice(i,1);up--}}if(allowAboveRoot){for(;up;up--){parts.unshift("..")}}return parts},normalize:path=>{var isAbsolute=PATH.isAbs(path),trailingSlash=path.slice(-1)==="/";path=PATH.normalizeArray(path.split("/").filter(p=>!!p),!isAbsolute).join("/");if(!path&&!isAbsolute){path="."}if(path&&trailingSlash){path+="/"}return(isAbsolute?"/":"")+path},dirname:path=>{var result=PATH.splitPath(path),root=result[0],dir=result[1];if(!root&&!dir){return"."}if(dir){dir=dir.slice(0,-1)}return root+dir},basename:path=>path&&path.match(/([^\\/]+|\\/)\\/*$/)[1],join:(...paths)=>PATH.normalize(paths.join("/")),join2:(l,r)=>PATH.normalize(l+"/"+r)};var initRandomFill=()=>view=>view.set(crypto.getRandomValues(new Uint8Array(view.byteLength)));var randomFill=view=>{(randomFill=initRandomFill())(view)};var PATH_FS={resolve:(...args)=>{var resolvedPath="",resolvedAbsolute=false;for(var i=args.length-1;i>=-1&&!resolvedAbsolute;i--){var path=i>=0?args[i]:FS.cwd();if(typeof path!="string"){throw new TypeError("Arguments to path.resolve must be strings")}else if(!path){return""}resolvedPath=path+"/"+resolvedPath;resolvedAbsolute=PATH.isAbs(path)}resolvedPath=PATH.normalizeArray(resolvedPath.split("/").filter(p=>!!p),!resolvedAbsolute).join("/");return(resolvedAbsolute?"/":"")+resolvedPath||"."},relative:(from,to)=>{from=PATH_FS.resolve(from).slice(1);to=PATH_FS.resolve(to).slice(1);function trim(arr){var start=0;for(;start<arr.length;start++){if(arr[start]!=="")break}var end=arr.length-1;for(;end>=0;end--){if(arr[end]!=="")break}if(start>end)return[];return arr.slice(start,end-start+1)}var fromParts=trim(from.split("/"));var toParts=trim(to.split("/"));var length=Math.min(fromParts.length,toParts.length);var samePartsLength=length;for(var i=0;i<length;i++){if(fromParts[i]!==toParts[i]){samePartsLength=i;break}}var outputParts=[];for(var i=samePartsLength;i<fromParts.length;i++){outputParts.push("..")}outputParts=outputParts.concat(toParts.slice(samePartsLength));return outputParts.join("/")}};var UTF8Decoder=globalThis.TextDecoder&&new TextDecoder;var findStringEnd=(heapOrArray,idx,maxBytesToRead,ignoreNul)=>{var maxIdx=idx+maxBytesToRead;if(ignoreNul)return maxIdx;while(heapOrArray[idx]&&!(idx>=maxIdx))++idx;return idx};var UTF8ArrayToString=(heapOrArray,idx=0,maxBytesToRead,ignoreNul)=>{var endPtr=findStringEnd(heapOrArray,idx,maxBytesToRead,ignoreNul);if(endPtr-idx>16&&heapOrArray.buffer&&UTF8Decoder){return UTF8Decoder.decode(heapOrArray.buffer instanceof ArrayBuffer?heapOrArray.subarray(idx,endPtr):heapOrArray.slice(idx,endPtr))}var str="";while(idx<endPtr){var u0=heapOrArray[idx++];if(!(u0&128)){str+=String.fromCharCode(u0);continue}var u1=heapOrArray[idx++]&63;if((u0&224)==192){str+=String.fromCharCode((u0&31)<<6|u1);continue}var u2=heapOrArray[idx++]&63;if((u0&240)==224){u0=(u0&15)<<12|u1<<6|u2}else{u0=(u0&7)<<18|u1<<12|u2<<6|heapOrArray[idx++]&63}if(u0<65536){str+=String.fromCharCode(u0)}else{var ch=u0-65536;str+=String.fromCharCode(55296|ch>>10,56320|ch&1023)}}return str};var FS_stdin_getChar_buffer=[];var lengthBytesUTF8=str=>{var len=0;for(var i=0;i<str.length;++i){var c=str.charCodeAt(i);if(c<=127){len++}else if(c<=2047){len+=2}else if(c>=55296&&c<=57343){len+=4;++i}else{len+=3}}return len};var stringToUTF8Array=(str,heap,outIdx,maxBytesToWrite)=>{if(!(maxBytesToWrite>0))return 0;var startIdx=outIdx;var endIdx=outIdx+maxBytesToWrite-1;for(var i=0;i<str.length;++i){var u=str.codePointAt(i);if(u<=127){if(outIdx>=endIdx)break;heap[outIdx++]=u}else if(u<=2047){if(outIdx+1>=endIdx)break;heap[outIdx++]=192|u>>6;heap[outIdx++]=128|u&63}else if(u<=65535){if(outIdx+2>=endIdx)break;heap[outIdx++]=224|u>>12;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63}else{if(outIdx+3>=endIdx)break;heap[outIdx++]=240|u>>18;heap[outIdx++]=128|u>>12&63;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63;i++}}heap[outIdx]=0;return outIdx-startIdx};var intArrayFromString=(stringy,dontAddNull,length)=>{var len=length>0?length:lengthBytesUTF8(stringy)+1;var u8array=new Array(len);var numBytesWritten=stringToUTF8Array(stringy,u8array,0,u8array.length);if(dontAddNull)u8array.length=numBytesWritten;return u8array};var FS_stdin_getChar=()=>{if(!FS_stdin_getChar_buffer.length){var result=null;if(ENVIRONMENT_IS_NODE){var BUFSIZE=256;var buf=Buffer.alloc(BUFSIZE);var bytesRead=0;var fd=process.stdin.fd;try{bytesRead=fs.readSync(fd,buf,0,BUFSIZE)}catch(e){if(e.toString().includes("EOF"))bytesRead=0;else throw e}if(bytesRead>0){result=buf.slice(0,bytesRead).toString("utf-8")}}else if(globalThis.window?.prompt){result=window.prompt("Input: ");if(result!==null){result+="\\n"}}else{}if(!result){return null}FS_stdin_getChar_buffer=intArrayFromString(result,true)}return FS_stdin_getChar_buffer.shift()};var TTY={ttys:[],init(){},shutdown(){},register(dev,ops){TTY.ttys[dev]={input:[],output:[],ops};FS.registerDevice(dev,TTY.stream_ops)},stream_ops:{open(stream){var tty=TTY.ttys[stream.node.rdev];if(!tty){throw new FS.ErrnoError(43)}stream.tty=tty;stream.seekable=false},close(stream){stream.tty.ops.fsync(stream.tty)},fsync(stream){stream.tty.ops.fsync(stream.tty)},read(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.get_char){throw new FS.ErrnoError(60)}var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=stream.tty.ops.get_char(stream.tty)}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.put_char){throw new FS.ErrnoError(60)}try{for(var i=0;i<length;i++){stream.tty.ops.put_char(stream.tty,buffer[offset+i])}}catch(e){throw new FS.ErrnoError(29)}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}},default_tty_ops:{get_char(tty){return FS_stdin_getChar()},put_char(tty,val){if(val===null||val===10){out(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){out(UTF8ArrayToString(tty.output));tty.output=[]}},ioctl_tcgets(tty){return{c_iflag:25856,c_oflag:5,c_cflag:191,c_lflag:35387,c_cc:[3,28,127,21,4,0,1,0,17,19,26,0,18,15,23,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}},ioctl_tcsets(tty,optional_actions,data){return 0},ioctl_tiocgwinsz(tty){return[24,80]}},default_tty1_ops:{put_char(tty,val){if(val===null||val===10){err(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){err(UTF8ArrayToString(tty.output));tty.output=[]}}}};var zeroMemory=(ptr,size)=>(growMemViews(),HEAPU8).fill(0,ptr,ptr+size);var alignMemory=(size,alignment)=>Math.ceil(size/alignment)*alignment;var mmapAlloc=size=>{size=alignMemory(size,65536);var ptr=_emscripten_builtin_memalign(65536,size);if(ptr)zeroMemory(ptr,size);return ptr};var MEMFS={ops_table:null,mount(mount){return MEMFS.createNode(null,"/",16895,0)},createNode(parent,name,mode,dev){if(FS.isBlkdev(mode)||FS.isFIFO(mode)){throw new FS.ErrnoError(63)}MEMFS.ops_table||={dir:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,lookup:MEMFS.node_ops.lookup,mknod:MEMFS.node_ops.mknod,rename:MEMFS.node_ops.rename,unlink:MEMFS.node_ops.unlink,rmdir:MEMFS.node_ops.rmdir,readdir:MEMFS.node_ops.readdir,symlink:MEMFS.node_ops.symlink},stream:{llseek:MEMFS.stream_ops.llseek}},file:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:{llseek:MEMFS.stream_ops.llseek,read:MEMFS.stream_ops.read,write:MEMFS.stream_ops.write,mmap:MEMFS.stream_ops.mmap,msync:MEMFS.stream_ops.msync}},link:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,readlink:MEMFS.node_ops.readlink},stream:{}},chrdev:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:FS.chrdev_stream_ops}};var node=FS.createNode(parent,name,mode,dev);if(FS.isDir(node.mode)){node.node_ops=MEMFS.ops_table.dir.node;node.stream_ops=MEMFS.ops_table.dir.stream;node.contents={}}else if(FS.isFile(node.mode)){node.node_ops=MEMFS.ops_table.file.node;node.stream_ops=MEMFS.ops_table.file.stream;node.usedBytes=0;node.contents=null}else if(FS.isLink(node.mode)){node.node_ops=MEMFS.ops_table.link.node;node.stream_ops=MEMFS.ops_table.link.stream}else if(FS.isChrdev(node.mode)){node.node_ops=MEMFS.ops_table.chrdev.node;node.stream_ops=MEMFS.ops_table.chrdev.stream}node.atime=node.mtime=node.ctime=Date.now();if(parent){parent.contents[name]=node;parent.atime=parent.mtime=parent.ctime=node.atime}return node},getFileDataAsTypedArray(node){if(!node.contents)return new Uint8Array(0);if(node.contents.subarray)return node.contents.subarray(0,node.usedBytes);return new Uint8Array(node.contents)},expandFileStorage(node,newCapacity){var prevCapacity=node.contents?node.contents.length:0;if(prevCapacity>=newCapacity)return;var CAPACITY_DOUBLING_MAX=1024*1024;newCapacity=Math.max(newCapacity,prevCapacity*(prevCapacity<CAPACITY_DOUBLING_MAX?2:1.125)>>>0);if(prevCapacity!=0)newCapacity=Math.max(newCapacity,256);var oldContents=node.contents;node.contents=new Uint8Array(newCapacity);if(node.usedBytes>0)node.contents.set(oldContents.subarray(0,node.usedBytes),0)},resizeFileStorage(node,newSize){if(node.usedBytes==newSize)return;if(newSize==0){node.contents=null;node.usedBytes=0}else{var oldContents=node.contents;node.contents=new Uint8Array(newSize);if(oldContents){node.contents.set(oldContents.subarray(0,Math.min(newSize,node.usedBytes)))}node.usedBytes=newSize}},node_ops:{getattr(node){var attr={};attr.dev=FS.isChrdev(node.mode)?node.id:1;attr.ino=node.id;attr.mode=node.mode;attr.nlink=1;attr.uid=0;attr.gid=0;attr.rdev=node.rdev;if(FS.isDir(node.mode)){attr.size=4096}else if(FS.isFile(node.mode)){attr.size=node.usedBytes}else if(FS.isLink(node.mode)){attr.size=node.link.length}else{attr.size=0}attr.atime=new Date(node.atime);attr.mtime=new Date(node.mtime);attr.ctime=new Date(node.ctime);attr.blksize=4096;attr.blocks=Math.ceil(attr.size/attr.blksize);return attr},setattr(node,attr){for(const key of["mode","atime","mtime","ctime"]){if(attr[key]!=null){node[key]=attr[key]}}if(attr.size!==undefined){MEMFS.resizeFileStorage(node,attr.size)}},lookup(parent,name){if(!MEMFS.doesNotExistError){MEMFS.doesNotExistError=new FS.ErrnoError(44);MEMFS.doesNotExistError.stack="<generic error, no stack>"}throw MEMFS.doesNotExistError},mknod(parent,name,mode,dev){return MEMFS.createNode(parent,name,mode,dev)},rename(old_node,new_dir,new_name){var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(new_node){if(FS.isDir(old_node.mode)){for(var i in new_node.contents){throw new FS.ErrnoError(55)}}FS.hashRemoveNode(new_node)}delete old_node.parent.contents[old_node.name];new_dir.contents[new_name]=old_node;old_node.name=new_name;new_dir.ctime=new_dir.mtime=old_node.parent.ctime=old_node.parent.mtime=Date.now()},unlink(parent,name){delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},rmdir(parent,name){var node=FS.lookupNode(parent,name);for(var i in node.contents){throw new FS.ErrnoError(55)}delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},readdir(node){return[".","..",...Object.keys(node.contents)]},symlink(parent,newname,oldpath){var node=MEMFS.createNode(parent,newname,511|40960,0);node.link=oldpath;return node},readlink(node){if(!FS.isLink(node.mode)){throw new FS.ErrnoError(28)}return node.link}},stream_ops:{read(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=stream.node.usedBytes)return 0;var size=Math.min(stream.node.usedBytes-position,length);if(size>8&&contents.subarray){buffer.set(contents.subarray(position,position+size),offset)}else{for(var i=0;i<size;i++)buffer[offset+i]=contents[position+i]}return size},write(stream,buffer,offset,length,position,canOwn){if(buffer.buffer===(growMemViews(),HEAP8).buffer){canOwn=false}if(!length)return 0;var node=stream.node;node.mtime=node.ctime=Date.now();if(buffer.subarray&&(!node.contents||node.contents.subarray)){if(canOwn){node.contents=buffer.subarray(offset,offset+length);node.usedBytes=length;return length}else if(node.usedBytes===0&&position===0){node.contents=buffer.slice(offset,offset+length);node.usedBytes=length;return length}else if(position+length<=node.usedBytes){node.contents.set(buffer.subarray(offset,offset+length),position);return length}}MEMFS.expandFileStorage(node,position+length);if(node.contents.subarray&&buffer.subarray){node.contents.set(buffer.subarray(offset,offset+length),position)}else{for(var i=0;i<length;i++){node.contents[position+i]=buffer[offset+i]}}node.usedBytes=Math.max(node.usedBytes,position+length);return length},llseek(stream,offset,whence){var position=offset;if(whence===1){position+=stream.position}else if(whence===2){if(FS.isFile(stream.node.mode)){position+=stream.node.usedBytes}}if(position<0){throw new FS.ErrnoError(28)}return position},mmap(stream,length,position,prot,flags){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}var ptr;var allocated;var contents=stream.node.contents;if(!(flags&2)&&contents&&contents.buffer===(growMemViews(),HEAP8).buffer){allocated=false;ptr=contents.byteOffset}else{allocated=true;ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}if(contents){if(position>0||position+length<contents.length){if(contents.subarray){contents=contents.subarray(position,position+length)}else{contents=Array.prototype.slice.call(contents,position,position+length)}}(growMemViews(),HEAP8).set(contents,ptr)}}return{ptr,allocated}},msync(stream,buffer,offset,length,mmapFlags){MEMFS.stream_ops.write(stream,buffer,0,length,offset,false);return 0}}};var FS_modeStringToFlags=str=>{var flagModes={r:0,"r+":2,w:512|64|1,"w+":512|64|2,a:1024|64|1,"a+":1024|64|2};var flags=flagModes[str];if(typeof flags=="undefined"){throw new Error(`Unknown file open mode: ${str}`)}return flags};var FS_getMode=(canRead,canWrite)=>{var mode=0;if(canRead)mode|=292|73;if(canWrite)mode|=146;return mode};var asyncLoad=async url=>{var arrayBuffer=await readAsync(url);return new Uint8Array(arrayBuffer)};var FS_createDataFile=(...args)=>FS.createDataFile(...args);var getUniqueRunDependency=id=>id;var preloadPlugins=[];var FS_handledByPreloadPlugin=async(byteArray,fullname)=>{if(typeof Browser!="undefined")Browser.init();for(var plugin of preloadPlugins){if(plugin["canHandle"](fullname)){return plugin["handle"](byteArray,fullname)}}return byteArray};var FS_preloadFile=async(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish)=>{var fullname=name?PATH_FS.resolve(PATH.join2(parent,name)):parent;var dep=getUniqueRunDependency(`cp ${fullname}`);addRunDependency(dep);try{var byteArray=url;if(typeof url=="string"){byteArray=await asyncLoad(url)}byteArray=await FS_handledByPreloadPlugin(byteArray,fullname);preFinish?.();if(!dontCreateFile){FS_createDataFile(parent,name,byteArray,canRead,canWrite,canOwn)}}finally{removeRunDependency(dep)}};var FS_createPreloadedFile=(parent,name,url,canRead,canWrite,onload,onerror,dontCreateFile,canOwn,preFinish)=>{FS_preloadFile(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish).then(onload).catch(onerror)};var FS={root:null,mounts:[],devices:{},streams:[],nextInode:1,nameTable:null,currentPath:"/",initialized:false,ignorePermissions:true,filesystems:null,syncFSRequests:0,readFiles:{},ErrnoError:class{name="ErrnoError";constructor(errno){this.errno=errno}},FSStream:class{shared={};get object(){return this.node}set object(val){this.node=val}get isRead(){return(this.flags&2097155)!==1}get isWrite(){return(this.flags&2097155)!==0}get isAppend(){return this.flags&1024}get flags(){return this.shared.flags}set flags(val){this.shared.flags=val}get position(){return this.shared.position}set position(val){this.shared.position=val}},FSNode:class{node_ops={};stream_ops={};readMode=292|73;writeMode=146;mounted=null;constructor(parent,name,mode,rdev){if(!parent){parent=this}this.parent=parent;this.mount=parent.mount;this.id=FS.nextInode++;this.name=name;this.mode=mode;this.rdev=rdev;this.atime=this.mtime=this.ctime=Date.now()}get read(){return(this.mode&this.readMode)===this.readMode}set read(val){val?this.mode|=this.readMode:this.mode&=~this.readMode}get write(){return(this.mode&this.writeMode)===this.writeMode}set write(val){val?this.mode|=this.writeMode:this.mode&=~this.writeMode}get isFolder(){return FS.isDir(this.mode)}get isDevice(){return FS.isChrdev(this.mode)}},lookupPath(path,opts={}){if(!path){throw new FS.ErrnoError(44)}opts.follow_mount??=true;if(!PATH.isAbs(path)){path=FS.cwd()+"/"+path}linkloop:for(var nlinks=0;nlinks<40;nlinks++){var parts=path.split("/").filter(p=>!!p);var current=FS.root;var current_path="/";for(var i=0;i<parts.length;i++){var islast=i===parts.length-1;if(islast&&opts.parent){break}if(parts[i]==="."){continue}if(parts[i]===".."){current_path=PATH.dirname(current_path);if(FS.isRoot(current)){path=current_path+"/"+parts.slice(i+1).join("/");nlinks--;continue linkloop}else{current=current.parent}continue}current_path=PATH.join2(current_path,parts[i]);try{current=FS.lookupNode(current,parts[i])}catch(e){if(e?.errno===44&&islast&&opts.noent_okay){return{path:current_path}}throw e}if(FS.isMountpoint(current)&&(!islast||opts.follow_mount)){current=current.mounted.root}if(FS.isLink(current.mode)&&(!islast||opts.follow)){if(!current.node_ops.readlink){throw new FS.ErrnoError(52)}var link=current.node_ops.readlink(current);if(!PATH.isAbs(link)){link=PATH.dirname(current_path)+"/"+link}path=link+"/"+parts.slice(i+1).join("/");continue linkloop}}return{path:current_path,node:current}}throw new FS.ErrnoError(32)},getPath(node){var path;while(true){if(FS.isRoot(node)){var mount=node.mount.mountpoint;if(!path)return mount;return mount[mount.length-1]!=="/"?`${mount}/${path}`:mount+path}path=path?`${node.name}/${path}`:node.name;node=node.parent}},hashName(parentid,name){var hash=0;for(var i=0;i<name.length;i++){hash=(hash<<5)-hash+name.charCodeAt(i)|0}return(parentid+hash>>>0)%FS.nameTable.length},hashAddNode(node){var hash=FS.hashName(node.parent.id,node.name);node.name_next=FS.nameTable[hash];FS.nameTable[hash]=node},hashRemoveNode(node){var hash=FS.hashName(node.parent.id,node.name);if(FS.nameTable[hash]===node){FS.nameTable[hash]=node.name_next}else{var current=FS.nameTable[hash];while(current){if(current.name_next===node){current.name_next=node.name_next;break}current=current.name_next}}},lookupNode(parent,name){var errCode=FS.mayLookup(parent);if(errCode){throw new FS.ErrnoError(errCode)}var hash=FS.hashName(parent.id,name);for(var node=FS.nameTable[hash];node;node=node.name_next){var nodeName=node.name;if(node.parent.id===parent.id&&nodeName===name){return node}}return FS.lookup(parent,name)},createNode(parent,name,mode,rdev){var node=new FS.FSNode(parent,name,mode,rdev);FS.hashAddNode(node);return node},destroyNode(node){FS.hashRemoveNode(node)},isRoot(node){return node===node.parent},isMountpoint(node){return!!node.mounted},isFile(mode){return(mode&61440)===32768},isDir(mode){return(mode&61440)===16384},isLink(mode){return(mode&61440)===40960},isChrdev(mode){return(mode&61440)===8192},isBlkdev(mode){return(mode&61440)===24576},isFIFO(mode){return(mode&61440)===4096},isSocket(mode){return(mode&49152)===49152},flagsToPermissionString(flag){var perms=["r","w","rw"][flag&3];if(flag&512){perms+="w"}return perms},nodePermissions(node,perms){if(FS.ignorePermissions){return 0}if(perms.includes("r")&&!(node.mode&292)){return 2}else if(perms.includes("w")&&!(node.mode&146)){return 2}else if(perms.includes("x")&&!(node.mode&73)){return 2}return 0},mayLookup(dir){if(!FS.isDir(dir.mode))return 54;var errCode=FS.nodePermissions(dir,"x");if(errCode)return errCode;if(!dir.node_ops.lookup)return 2;return 0},mayCreate(dir,name){if(!FS.isDir(dir.mode)){return 54}try{var node=FS.lookupNode(dir,name);return 20}catch(e){}return FS.nodePermissions(dir,"wx")},mayDelete(dir,name,isdir){var node;try{node=FS.lookupNode(dir,name)}catch(e){return e.errno}var errCode=FS.nodePermissions(dir,"wx");if(errCode){return errCode}if(isdir){if(!FS.isDir(node.mode)){return 54}if(FS.isRoot(node)||FS.getPath(node)===FS.cwd()){return 10}}else{if(FS.isDir(node.mode)){return 31}}return 0},mayOpen(node,flags){if(!node){return 44}if(FS.isLink(node.mode)){return 32}else if(FS.isDir(node.mode)){if(FS.flagsToPermissionString(flags)!=="r"||flags&(512|64)){return 31}}return FS.nodePermissions(node,FS.flagsToPermissionString(flags))},checkOpExists(op,err){if(!op){throw new FS.ErrnoError(err)}return op},MAX_OPEN_FDS:4096,nextfd(){for(var fd=0;fd<=FS.MAX_OPEN_FDS;fd++){if(!FS.streams[fd]){return fd}}throw new FS.ErrnoError(33)},getStreamChecked(fd){var stream=FS.getStream(fd);if(!stream){throw new FS.ErrnoError(8)}return stream},getStream:fd=>FS.streams[fd],createStream(stream,fd=-1){stream=Object.assign(new FS.FSStream,stream);if(fd==-1){fd=FS.nextfd()}stream.fd=fd;FS.streams[fd]=stream;return stream},closeStream(fd){FS.streams[fd]=null},dupStream(origStream,fd=-1){var stream=FS.createStream(origStream,fd);stream.stream_ops?.dup?.(stream);return stream},doSetAttr(stream,node,attr){var setattr=stream?.stream_ops.setattr;var arg=setattr?stream:node;setattr??=node.node_ops.setattr;FS.checkOpExists(setattr,63);setattr(arg,attr)},chrdev_stream_ops:{open(stream){var device=FS.getDevice(stream.node.rdev);stream.stream_ops=device.stream_ops;stream.stream_ops.open?.(stream)},llseek(){throw new FS.ErrnoError(70)}},major:dev=>dev>>8,minor:dev=>dev&255,makedev:(ma,mi)=>ma<<8|mi,registerDevice(dev,ops){FS.devices[dev]={stream_ops:ops}},getDevice:dev=>FS.devices[dev],getMounts(mount){var mounts=[];var check=[mount];while(check.length){var m=check.pop();mounts.push(m);check.push(...m.mounts)}return mounts},syncfs(populate,callback){if(typeof populate=="function"){callback=populate;populate=false}FS.syncFSRequests++;if(FS.syncFSRequests>1){err(`warning: ${FS.syncFSRequests} FS.syncfs operations in flight at once, probably just doing extra work`)}var mounts=FS.getMounts(FS.root.mount);var completed=0;function doCallback(errCode){FS.syncFSRequests--;return callback(errCode)}function done(errCode){if(errCode){if(!done.errored){done.errored=true;return doCallback(errCode)}return}if(++completed>=mounts.length){doCallback(null)}}for(var mount of mounts){if(mount.type.syncfs){mount.type.syncfs(mount,populate,done)}else{done(null)}}},mount(type,opts,mountpoint){var root=mountpoint==="/";var pseudo=!mountpoint;var node;if(root&&FS.root){throw new FS.ErrnoError(10)}else if(!root&&!pseudo){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});mountpoint=lookup.path;node=lookup.node;if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}if(!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}}var mount={type,opts,mountpoint,mounts:[]};var mountRoot=type.mount(mount);mountRoot.mount=mount;mount.root=mountRoot;if(root){FS.root=mountRoot}else if(node){node.mounted=mount;if(node.mount){node.mount.mounts.push(mount)}}return mountRoot},unmount(mountpoint){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});if(!FS.isMountpoint(lookup.node)){throw new FS.ErrnoError(28)}var node=lookup.node;var mount=node.mounted;var mounts=FS.getMounts(mount);for(var[hash,current]of Object.entries(FS.nameTable)){while(current){var next=current.name_next;if(mounts.includes(current.mount)){FS.destroyNode(current)}current=next}}node.mounted=null;var idx=node.mount.mounts.indexOf(mount);node.mount.mounts.splice(idx,1)},lookup(parent,name){return parent.node_ops.lookup(parent,name)},mknod(path,mode,dev){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);if(!name){throw new FS.ErrnoError(28)}if(name==="."||name===".."){throw new FS.ErrnoError(20)}var errCode=FS.mayCreate(parent,name);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.mknod){throw new FS.ErrnoError(63)}return parent.node_ops.mknod(parent,name,mode,dev)},statfs(path){return FS.statfsNode(FS.lookupPath(path,{follow:true}).node)},statfsStream(stream){return FS.statfsNode(stream.node)},statfsNode(node){var rtn={bsize:4096,frsize:4096,blocks:1e6,bfree:5e5,bavail:5e5,files:FS.nextInode,ffree:FS.nextInode-1,fsid:42,flags:2,namelen:255};if(node.node_ops.statfs){Object.assign(rtn,node.node_ops.statfs(node.mount.opts.root))}return rtn},create(path,mode=438){mode&=4095;mode|=32768;return FS.mknod(path,mode,0)},mkdir(path,mode=511){mode&=511|512;mode|=16384;return FS.mknod(path,mode,0)},mkdirTree(path,mode){var dirs=path.split("/");var d="";for(var dir of dirs){if(!dir)continue;if(d||PATH.isAbs(path))d+="/";d+=dir;try{FS.mkdir(d,mode)}catch(e){if(e.errno!=20)throw e}}},mkdev(path,mode,dev){if(typeof dev=="undefined"){dev=mode;mode=438}mode|=8192;return FS.mknod(path,mode,dev)},symlink(oldpath,newpath){if(!PATH_FS.resolve(oldpath)){throw new FS.ErrnoError(44)}var lookup=FS.lookupPath(newpath,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var newname=PATH.basename(newpath);var errCode=FS.mayCreate(parent,newname);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.symlink){throw new FS.ErrnoError(63)}return parent.node_ops.symlink(parent,newname,oldpath)},rename(old_path,new_path){var old_dirname=PATH.dirname(old_path);var new_dirname=PATH.dirname(new_path);var old_name=PATH.basename(old_path);var new_name=PATH.basename(new_path);var lookup,old_dir,new_dir;lookup=FS.lookupPath(old_path,{parent:true});old_dir=lookup.node;lookup=FS.lookupPath(new_path,{parent:true});new_dir=lookup.node;if(!old_dir||!new_dir)throw new FS.ErrnoError(44);if(old_dir.mount!==new_dir.mount){throw new FS.ErrnoError(75)}var old_node=FS.lookupNode(old_dir,old_name);var relative=PATH_FS.relative(old_path,new_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(28)}relative=PATH_FS.relative(new_path,old_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(55)}var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(old_node===new_node){return}var isdir=FS.isDir(old_node.mode);var errCode=FS.mayDelete(old_dir,old_name,isdir);if(errCode){throw new FS.ErrnoError(errCode)}errCode=new_node?FS.mayDelete(new_dir,new_name,isdir):FS.mayCreate(new_dir,new_name);if(errCode){throw new FS.ErrnoError(errCode)}if(!old_dir.node_ops.rename){throw new FS.ErrnoError(63)}if(FS.isMountpoint(old_node)||new_node&&FS.isMountpoint(new_node)){throw new FS.ErrnoError(10)}if(new_dir!==old_dir){errCode=FS.nodePermissions(old_dir,"w");if(errCode){throw new FS.ErrnoError(errCode)}}FS.hashRemoveNode(old_node);try{old_dir.node_ops.rename(old_node,new_dir,new_name);old_node.parent=new_dir}catch(e){throw e}finally{FS.hashAddNode(old_node)}},rmdir(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,true);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.rmdir){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.rmdir(parent,name);FS.destroyNode(node)},readdir(path){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var readdir=FS.checkOpExists(node.node_ops.readdir,54);return readdir(node)},unlink(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,false);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.unlink){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.unlink(parent,name);FS.destroyNode(node)},readlink(path){var lookup=FS.lookupPath(path);var link=lookup.node;if(!link){throw new FS.ErrnoError(44)}if(!link.node_ops.readlink){throw new FS.ErrnoError(28)}return link.node_ops.readlink(link)},stat(path,dontFollow){var lookup=FS.lookupPath(path,{follow:!dontFollow});var node=lookup.node;var getattr=FS.checkOpExists(node.node_ops.getattr,63);return getattr(node)},fstat(fd){var stream=FS.getStreamChecked(fd);var node=stream.node;var getattr=stream.stream_ops.getattr;var arg=getattr?stream:node;getattr??=node.node_ops.getattr;FS.checkOpExists(getattr,63);return getattr(arg)},lstat(path){return FS.stat(path,true)},doChmod(stream,node,mode,dontFollow){FS.doSetAttr(stream,node,{mode:mode&4095|node.mode&~4095,ctime:Date.now(),dontFollow})},chmod(path,mode,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChmod(null,node,mode,dontFollow)},lchmod(path,mode){FS.chmod(path,mode,true)},fchmod(fd,mode){var stream=FS.getStreamChecked(fd);FS.doChmod(stream,stream.node,mode,false)},doChown(stream,node,dontFollow){FS.doSetAttr(stream,node,{timestamp:Date.now(),dontFollow})},chown(path,uid,gid,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChown(null,node,dontFollow)},lchown(path,uid,gid){FS.chown(path,uid,gid,true)},fchown(fd,uid,gid){var stream=FS.getStreamChecked(fd);FS.doChown(stream,stream.node,false)},doTruncate(stream,node,len){if(FS.isDir(node.mode)){throw new FS.ErrnoError(31)}if(!FS.isFile(node.mode)){throw new FS.ErrnoError(28)}var errCode=FS.nodePermissions(node,"w");if(errCode){throw new FS.ErrnoError(errCode)}FS.doSetAttr(stream,node,{size:len,timestamp:Date.now()})},truncate(path,len){if(len<0){throw new FS.ErrnoError(28)}var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:true});node=lookup.node}else{node=path}FS.doTruncate(null,node,len)},ftruncate(fd,len){var stream=FS.getStreamChecked(fd);if(len<0||(stream.flags&2097155)===0){throw new FS.ErrnoError(28)}FS.doTruncate(stream,stream.node,len)},utime(path,atime,mtime){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var setattr=FS.checkOpExists(node.node_ops.setattr,63);setattr(node,{atime,mtime})},open(path,flags,mode=438){if(path===""){throw new FS.ErrnoError(44)}flags=typeof flags=="string"?FS_modeStringToFlags(flags):flags;if(flags&64){mode=mode&4095|32768}else{mode=0}var node;var isDirPath;if(typeof path=="object"){node=path}else{isDirPath=path.endsWith("/");var lookup=FS.lookupPath(path,{follow:!(flags&131072),noent_okay:true});node=lookup.node;path=lookup.path}var created=false;if(flags&64){if(node){if(flags&128){throw new FS.ErrnoError(20)}}else if(isDirPath){throw new FS.ErrnoError(31)}else{node=FS.mknod(path,mode|511,0);created=true}}if(!node){throw new FS.ErrnoError(44)}if(FS.isChrdev(node.mode)){flags&=~512}if(flags&65536&&!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}if(!created){var errCode=FS.mayOpen(node,flags);if(errCode){throw new FS.ErrnoError(errCode)}}if(flags&512&&!created){FS.truncate(node,0)}flags&=~(128|512|131072);var stream=FS.createStream({node,path:FS.getPath(node),flags,seekable:true,position:0,stream_ops:node.stream_ops,ungotten:[],error:false});if(stream.stream_ops.open){stream.stream_ops.open(stream)}if(created){FS.chmod(node,mode&511)}if(Module["logReadFiles"]&&!(flags&1)){if(!(path in FS.readFiles)){FS.readFiles[path]=1}}return stream},close(stream){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(stream.getdents)stream.getdents=null;try{if(stream.stream_ops.close){stream.stream_ops.close(stream)}}catch(e){throw e}finally{FS.closeStream(stream.fd)}stream.fd=null},isClosed(stream){return stream.fd===null},llseek(stream,offset,whence){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(!stream.seekable||!stream.stream_ops.llseek){throw new FS.ErrnoError(70)}if(whence!=0&&whence!=1&&whence!=2){throw new FS.ErrnoError(28)}stream.position=stream.stream_ops.llseek(stream,offset,whence);stream.ungotten=[];return stream.position},read(stream,buffer,offset,length,position){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.read){throw new FS.ErrnoError(28)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesRead=stream.stream_ops.read(stream,buffer,offset,length,position);if(!seeking)stream.position+=bytesRead;return bytesRead},write(stream,buffer,offset,length,position,canOwn){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===0){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.write){throw new FS.ErrnoError(28)}if(stream.seekable&&stream.flags&1024){FS.llseek(stream,0,2)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesWritten=stream.stream_ops.write(stream,buffer,offset,length,position,canOwn);if(!seeking)stream.position+=bytesWritten;return bytesWritten},mmap(stream,length,position,prot,flags){if((prot&2)!==0&&(flags&2)===0&&(stream.flags&2097155)!==2){throw new FS.ErrnoError(2)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(2)}if(!stream.stream_ops.mmap){throw new FS.ErrnoError(43)}if(!length){throw new FS.ErrnoError(28)}return stream.stream_ops.mmap(stream,length,position,prot,flags)},msync(stream,buffer,offset,length,mmapFlags){if(!stream.stream_ops.msync){return 0}return stream.stream_ops.msync(stream,buffer,offset,length,mmapFlags)},ioctl(stream,cmd,arg){if(!stream.stream_ops.ioctl){throw new FS.ErrnoError(59)}return stream.stream_ops.ioctl(stream,cmd,arg)},readFile(path,opts={}){opts.flags=opts.flags||0;opts.encoding=opts.encoding||"binary";if(opts.encoding!=="utf8"&&opts.encoding!=="binary"){abort(`Invalid encoding type "${opts.encoding}"`)}var stream=FS.open(path,opts.flags);var stat=FS.stat(path);var length=stat.size;var buf=new Uint8Array(length);FS.read(stream,buf,0,length,0);if(opts.encoding==="utf8"){buf=UTF8ArrayToString(buf)}FS.close(stream);return buf},writeFile(path,data,opts={}){opts.flags=opts.flags||577;var stream=FS.open(path,opts.flags,opts.mode);if(typeof data=="string"){data=new Uint8Array(intArrayFromString(data,true))}if(ArrayBuffer.isView(data)){FS.write(stream,data,0,data.byteLength,undefined,opts.canOwn)}else{abort("Unsupported data type")}FS.close(stream)},cwd:()=>FS.currentPath,chdir(path){var lookup=FS.lookupPath(path,{follow:true});if(lookup.node===null){throw new FS.ErrnoError(44)}if(!FS.isDir(lookup.node.mode)){throw new FS.ErrnoError(54)}var errCode=FS.nodePermissions(lookup.node,"x");if(errCode){throw new FS.ErrnoError(errCode)}FS.currentPath=lookup.path},createDefaultDirectories(){FS.mkdir("/tmp");FS.mkdir("/home");FS.mkdir("/home/web_user")},createDefaultDevices(){FS.mkdir("/dev");FS.registerDevice(FS.makedev(1,3),{read:()=>0,write:(stream,buffer,offset,length,pos)=>length,llseek:()=>0});FS.mkdev("/dev/null",FS.makedev(1,3));TTY.register(FS.makedev(5,0),TTY.default_tty_ops);TTY.register(FS.makedev(6,0),TTY.default_tty1_ops);FS.mkdev("/dev/tty",FS.makedev(5,0));FS.mkdev("/dev/tty1",FS.makedev(6,0));var randomBuffer=new Uint8Array(1024),randomLeft=0;var randomByte=()=>{if(randomLeft===0){randomFill(randomBuffer);randomLeft=randomBuffer.byteLength}return randomBuffer[--randomLeft]};FS.createDevice("/dev","random",randomByte);FS.createDevice("/dev","urandom",randomByte);FS.mkdir("/dev/shm");FS.mkdir("/dev/shm/tmp")},createSpecialDirectories(){FS.mkdir("/proc");var proc_self=FS.mkdir("/proc/self");FS.mkdir("/proc/self/fd");FS.mount({mount(){var node=FS.createNode(proc_self,"fd",16895,73);node.stream_ops={llseek:MEMFS.stream_ops.llseek};node.node_ops={lookup(parent,name){var fd=+name;var stream=FS.getStreamChecked(fd);var ret={parent:null,mount:{mountpoint:"fake"},node_ops:{readlink:()=>stream.path},id:fd+1};ret.parent=ret;return ret},readdir(){return Array.from(FS.streams.entries()).filter(([k,v])=>v).map(([k,v])=>k.toString())}};return node}},{},"/proc/self/fd")},createStandardStreams(input,output,error){if(input){FS.createDevice("/dev","stdin",input)}else{FS.symlink("/dev/tty","/dev/stdin")}if(output){FS.createDevice("/dev","stdout",null,output)}else{FS.symlink("/dev/tty","/dev/stdout")}if(error){FS.createDevice("/dev","stderr",null,error)}else{FS.symlink("/dev/tty1","/dev/stderr")}var stdin=FS.open("/dev/stdin",0);var stdout=FS.open("/dev/stdout",1);var stderr=FS.open("/dev/stderr",1)},staticInit(){FS.nameTable=new Array(4096);FS.mount(MEMFS,{},"/");FS.createDefaultDirectories();FS.createDefaultDevices();FS.createSpecialDirectories();FS.filesystems={MEMFS}},init(input,output,error){FS.initialized=true;input??=Module["stdin"];output??=Module["stdout"];error??=Module["stderr"];FS.createStandardStreams(input,output,error)},quit(){FS.initialized=false;for(var stream of FS.streams){if(stream){FS.close(stream)}}},findObject(path,dontResolveLastLink){var ret=FS.analyzePath(path,dontResolveLastLink);if(!ret.exists){return null}return ret.object},analyzePath(path,dontResolveLastLink){try{var lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});path=lookup.path}catch(e){}var ret={isRoot:false,exists:false,error:0,name:null,path:null,object:null,parentExists:false,parentPath:null,parentObject:null};try{var lookup=FS.lookupPath(path,{parent:true});ret.parentExists=true;ret.parentPath=lookup.path;ret.parentObject=lookup.node;ret.name=PATH.basename(path);lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});ret.exists=true;ret.path=lookup.path;ret.object=lookup.node;ret.name=lookup.node.name;ret.isRoot=lookup.path==="/"}catch(e){ret.error=e.errno}return ret},createPath(parent,path,canRead,canWrite){parent=typeof parent=="string"?parent:FS.getPath(parent);var parts=path.split("/").reverse();while(parts.length){var part=parts.pop();if(!part)continue;var current=PATH.join2(parent,part);try{FS.mkdir(current)}catch(e){if(e.errno!=20)throw e}parent=current}return current},createFile(parent,name,properties,canRead,canWrite){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(canRead,canWrite);return FS.create(path,mode)},createDataFile(parent,name,data,canRead,canWrite,canOwn){var path=name;if(parent){parent=typeof parent=="string"?parent:FS.getPath(parent);path=name?PATH.join2(parent,name):parent}var mode=FS_getMode(canRead,canWrite);var node=FS.create(path,mode);if(data){if(typeof data=="string"){var arr=new Array(data.length);for(var i=0,len=data.length;i<len;++i)arr[i]=data.charCodeAt(i);data=arr}FS.chmod(node,mode|146);var stream=FS.open(node,577);FS.write(stream,data,0,data.length,0,canOwn);FS.close(stream);FS.chmod(node,mode)}},createDevice(parent,name,input,output){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(!!input,!!output);FS.createDevice.major??=64;var dev=FS.makedev(FS.createDevice.major++,0);FS.registerDevice(dev,{open(stream){stream.seekable=false},close(stream){if(output?.buffer?.length){output(10)}},read(stream,buffer,offset,length,pos){var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=input()}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){for(var i=0;i<length;i++){try{output(buffer[offset+i])}catch(e){throw new FS.ErrnoError(29)}}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}});return FS.mkdev(path,mode,dev)},forceLoadFile(obj){if(obj.isDevice||obj.isFolder||obj.link||obj.contents)return true;if(globalThis.XMLHttpRequest){abort("Lazy loading should have been performed (contents set) in createLazyFile, but it was not. Lazy loading only works in web workers. Use --embed-file or --preload-file in emcc on the main thread.")}else{try{obj.contents=readBinary(obj.url)}catch(e){throw new FS.ErrnoError(29)}}},createLazyFile(parent,name,url,canRead,canWrite){class LazyUint8Array{lengthKnown=false;chunks=[];get(idx){if(idx>this.length-1||idx<0){return undefined}var chunkOffset=idx%this.chunkSize;var chunkNum=idx/this.chunkSize|0;return this.getter(chunkNum)[chunkOffset]}setDataGetter(getter){this.getter=getter}cacheLength(){var xhr=new XMLHttpRequest;xhr.open("HEAD",url,false);xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn\'t load "+url+". Status: "+xhr.status);var datalength=Number(xhr.getResponseHeader("Content-length"));var header;var hasByteServing=(header=xhr.getResponseHeader("Accept-Ranges"))&&header==="bytes";var usesGzip=(header=xhr.getResponseHeader("Content-Encoding"))&&header==="gzip";var chunkSize=1024*1024;if(!hasByteServing)chunkSize=datalength;var doXHR=(from,to)=>{if(from>to)abort("invalid range ("+from+", "+to+") or no bytes requested!");if(to>datalength-1)abort("only "+datalength+" bytes available! programmer error!");var xhr=new XMLHttpRequest;xhr.open("GET",url,false);if(datalength!==chunkSize)xhr.setRequestHeader("Range","bytes="+from+"-"+to);xhr.responseType="arraybuffer";if(xhr.overrideMimeType){xhr.overrideMimeType("text/plain; charset=x-user-defined")}xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn\'t load "+url+". Status: "+xhr.status);if(xhr.response!==undefined){return new Uint8Array(xhr.response||[])}return intArrayFromString(xhr.responseText||"",true)};var lazyArray=this;lazyArray.setDataGetter(chunkNum=>{var start=chunkNum*chunkSize;var end=(chunkNum+1)*chunkSize-1;end=Math.min(end,datalength-1);if(typeof lazyArray.chunks[chunkNum]=="undefined"){lazyArray.chunks[chunkNum]=doXHR(start,end)}if(typeof lazyArray.chunks[chunkNum]=="undefined")abort("doXHR failed!");return lazyArray.chunks[chunkNum]});if(usesGzip||!datalength){chunkSize=datalength=1;datalength=this.getter(0).length;chunkSize=datalength;out("LazyFiles on gzip forces download of the whole file when length is accessed")}this._length=datalength;this._chunkSize=chunkSize;this.lengthKnown=true}get length(){if(!this.lengthKnown){this.cacheLength()}return this._length}get chunkSize(){if(!this.lengthKnown){this.cacheLength()}return this._chunkSize}}if(globalThis.XMLHttpRequest){if(!ENVIRONMENT_IS_WORKER)abort("Cannot do synchronous binary XHRs outside webworkers in modern browsers. Use --embed-file or --preload-file in emcc");var lazyArray=new LazyUint8Array;var properties={isDevice:false,contents:lazyArray}}else{var properties={isDevice:false,url}}var node=FS.createFile(parent,name,properties,canRead,canWrite);if(properties.contents){node.contents=properties.contents}else if(properties.url){node.contents=null;node.url=properties.url}Object.defineProperties(node,{usedBytes:{get:function(){return this.contents.length}}});var stream_ops={};for(const[key,fn]of Object.entries(node.stream_ops)){stream_ops[key]=(...args)=>{FS.forceLoadFile(node);return fn(...args)}}function writeChunks(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=contents.length)return 0;var size=Math.min(contents.length-position,length);if(contents.slice){for(var i=0;i<size;i++){buffer[offset+i]=contents[position+i]}}else{for(var i=0;i<size;i++){buffer[offset+i]=contents.get(position+i)}}return size}stream_ops.read=(stream,buffer,offset,length,position)=>{FS.forceLoadFile(node);return writeChunks(stream,buffer,offset,length,position)};stream_ops.mmap=(stream,length,position,prot,flags)=>{FS.forceLoadFile(node);var ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}writeChunks(stream,(growMemViews(),HEAP8),ptr,length,position);return{ptr,allocated:true}};node.stream_ops=stream_ops;return node}};var UTF8ToString=(ptr,maxBytesToRead,ignoreNul)=>ptr?UTF8ArrayToString((growMemViews(),HEAPU8),ptr,maxBytesToRead,ignoreNul):"";var SYSCALLS={DEFAULT_POLLMASK:5,calculateAt(dirfd,path,allowEmpty){if(PATH.isAbs(path)){return path}var dir;if(dirfd===-100){dir=FS.cwd()}else{var dirstream=SYSCALLS.getStreamFromFD(dirfd);dir=dirstream.path}if(path.length==0){if(!allowEmpty){throw new FS.ErrnoError(44)}return dir}return dir+"/"+path},writeStat(buf,stat){(growMemViews(),HEAPU32)[buf/4]=stat.dev;(growMemViews(),HEAPU32)[(buf+4)/4]=stat.mode;(growMemViews(),HEAPU64)[(buf+8)/8]=BigInt(stat.nlink);(growMemViews(),HEAPU32)[(buf+16)/4]=stat.uid;(growMemViews(),HEAPU32)[(buf+20)/4]=stat.gid;(growMemViews(),HEAPU32)[(buf+24)/4]=stat.rdev;(growMemViews(),HEAP64)[(buf+32)/8]=BigInt(stat.size);(growMemViews(),HEAP32)[(buf+40)/4]=4096;(growMemViews(),HEAP32)[(buf+44)/4]=stat.blocks;var atime=stat.atime.getTime();var mtime=stat.mtime.getTime();var ctime=stat.ctime.getTime();(growMemViews(),HEAP64)[(buf+48)/8]=BigInt(Math.floor(atime/1e3));(growMemViews(),HEAPU64)[(buf+56)/8]=BigInt(atime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+64)/8]=BigInt(Math.floor(mtime/1e3));(growMemViews(),HEAPU64)[(buf+72)/8]=BigInt(mtime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+80)/8]=BigInt(Math.floor(ctime/1e3));(growMemViews(),HEAPU64)[(buf+88)/8]=BigInt(ctime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+96)/8]=BigInt(stat.ino);return 0},writeStatFs(buf,stats){(growMemViews(),HEAPU32)[(buf+8)/4]=stats.bsize;(growMemViews(),HEAPU32)[(buf+72)/4]=stats.bsize;(growMemViews(),HEAP64)[(buf+16)/8]=BigInt(stats.blocks);(growMemViews(),HEAP64)[(buf+24)/8]=BigInt(stats.bfree);(growMemViews(),HEAP64)[(buf+32)/8]=BigInt(stats.bavail);(growMemViews(),HEAP64)[(buf+40)/8]=BigInt(stats.files);(growMemViews(),HEAP64)[(buf+48)/8]=BigInt(stats.ffree);(growMemViews(),HEAPU32)[(buf+56)/4]=stats.fsid;(growMemViews(),HEAPU32)[(buf+80)/4]=stats.flags;(growMemViews(),HEAPU32)[(buf+64)/4]=stats.namelen},doMsync(addr,stream,len,flags,offset){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}if(flags&2){return 0}var buffer=(growMemViews(),HEAPU8).slice(addr,addr+len);FS.msync(stream,buffer,offset,len,flags)},getStreamFromFD(fd){var stream=FS.getStreamChecked(fd);return stream},varargs:undefined,getStr(ptr){var ret=UTF8ToString(ptr);return ret}};function ___syscall_fcntl64(fd,cmd,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(3,0,1,fd,cmd,varargs);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(cmd){case 0:{var arg=syscallGetVarargI();if(arg<0){return-28}while(FS.streams[arg]){arg++}var newStream;newStream=FS.dupStream(stream,arg);return newStream.fd}case 1:case 2:return 0;case 3:return stream.flags;case 4:{var arg=syscallGetVarargI();stream.flags|=arg;return 0}case 5:{var arg=syscallGetVarargP();var offset=0;(growMemViews(),HEAP16)[(arg+offset)/2]=2;return 0}case 6:case 7:return 0}return-28}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_fstat64(fd,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(4,0,1,fd,buf);buf=bigintToI53Checked(buf);try{return SYSCALLS.writeStat(buf,FS.fstat(fd))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var stringToUTF8=(str,outPtr,maxBytesToWrite)=>stringToUTF8Array(str,(growMemViews(),HEAPU8),outPtr,maxBytesToWrite);function ___syscall_getcwd(buf,size){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(5,0,1,buf,size);buf=bigintToI53Checked(buf);size=bigintToI53Checked(size);try{if(size===0)return-28;var cwd=FS.cwd();var cwdLengthInBytes=lengthBytesUTF8(cwd)+1;if(size<cwdLengthInBytes)return-68;stringToUTF8(cwd,buf,size);return cwdLengthInBytes}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_getdents64(fd,dirp,count){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(6,0,1,fd,dirp,count);dirp=bigintToI53Checked(dirp);count=bigintToI53Checked(count);try{var stream=SYSCALLS.getStreamFromFD(fd);stream.getdents||=FS.readdir(stream.path);var struct_size=280;var pos=0;var off=FS.llseek(stream,0,1);var startIdx=Math.floor(off/struct_size);var endIdx=Math.min(stream.getdents.length,startIdx+Math.floor(count/struct_size));for(var idx=startIdx;idx<endIdx;idx++){var id;var type;var name=stream.getdents[idx];if(name==="."){id=stream.node.id;type=4}else if(name===".."){var lookup=FS.lookupPath(stream.path,{parent:true});id=lookup.node.id;type=4}else{var child;try{child=FS.lookupNode(stream.node,name)}catch(e){if(e?.errno===28){continue}throw e}id=child.id;type=FS.isChrdev(child.mode)?2:FS.isDir(child.mode)?4:FS.isLink(child.mode)?10:8}(growMemViews(),HEAP64)[(dirp+pos)/8]=BigInt(id);(growMemViews(),HEAP64)[(dirp+pos+8)/8]=BigInt((idx+1)*struct_size);(growMemViews(),HEAP16)[(dirp+pos+16)/2]=280;(growMemViews(),HEAP8)[dirp+pos+18]=type;stringToUTF8(name,dirp+pos+19,256);pos+=struct_size}FS.llseek(stream,idx*struct_size,0);return pos}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_ioctl(fd,op,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(7,0,1,fd,op,varargs);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(op){case 21509:{if(!stream.tty)return-59;return 0}case 21505:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcgets){var termios=stream.tty.ops.ioctl_tcgets(stream);var argp=syscallGetVarargP();(growMemViews(),HEAP32)[argp/4]=termios.c_iflag||0;(growMemViews(),HEAP32)[(argp+4)/4]=termios.c_oflag||0;(growMemViews(),HEAP32)[(argp+8)/4]=termios.c_cflag||0;(growMemViews(),HEAP32)[(argp+12)/4]=termios.c_lflag||0;for(var i=0;i<32;i++){(growMemViews(),HEAP8)[argp+i+17]=termios.c_cc[i]||0}return 0}return 0}case 21510:case 21511:case 21512:{if(!stream.tty)return-59;return 0}case 21506:case 21507:case 21508:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcsets){var argp=syscallGetVarargP();var c_iflag=(growMemViews(),HEAP32)[argp/4];var c_oflag=(growMemViews(),HEAP32)[(argp+4)/4];var c_cflag=(growMemViews(),HEAP32)[(argp+8)/4];var c_lflag=(growMemViews(),HEAP32)[(argp+12)/4];var c_cc=[];for(var i=0;i<32;i++){c_cc.push((growMemViews(),HEAP8)[argp+i+17])}return stream.tty.ops.ioctl_tcsets(stream.tty,op,{c_iflag,c_oflag,c_cflag,c_lflag,c_cc})}return 0}case 21519:{if(!stream.tty)return-59;var argp=syscallGetVarargP();(growMemViews(),HEAP32)[argp/4]=0;return 0}case 21520:{if(!stream.tty)return-59;return-28}case 21537:case 21531:{var argp=syscallGetVarargP();return FS.ioctl(stream,op,argp)}case 21523:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tiocgwinsz){var winsize=stream.tty.ops.ioctl_tiocgwinsz(stream.tty);var argp=syscallGetVarargP();(growMemViews(),HEAP16)[argp/2]=winsize[0];(growMemViews(),HEAP16)[(argp+2)/2]=winsize[1]}return 0}case 21524:{if(!stream.tty)return-59;return 0}case 21515:{if(!stream.tty)return-59;return 0}default:return-28}}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_lstat64(path,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(8,0,1,path,buf);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);return SYSCALLS.writeStat(buf,FS.lstat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_newfstatat(dirfd,path,buf,flags){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(9,0,1,dirfd,path,buf,flags);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);var nofollow=flags&256;var allowEmpty=flags&4096;flags=flags&~6400;path=SYSCALLS.calculateAt(dirfd,path,allowEmpty);return SYSCALLS.writeStat(buf,nofollow?FS.lstat(path):FS.stat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_openat(dirfd,path,flags,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(10,0,1,dirfd,path,flags,varargs);path=bigintToI53Checked(path);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{path=SYSCALLS.getStr(path);path=SYSCALLS.calculateAt(dirfd,path);var mode=varargs?syscallGetVarargI():0;return FS.open(path,flags,mode).fd}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_stat64(path,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(11,0,1,path,buf);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);return SYSCALLS.writeStat(buf,FS.stat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var __abort_js=()=>abort("");function __emscripten_init_main_thread_js(tb){tb=bigintToI53Checked(tb);__emscripten_thread_init(tb,!ENVIRONMENT_IS_WORKER,1,!ENVIRONMENT_IS_WEB,5242880,false);PThread.threadInitTLS()}var handleException=e=>{if(e instanceof ExitStatus||e=="unwind"){return EXITSTATUS}quit_(1,e)};var maybeExit=()=>{if(!keepRuntimeAlive()){try{if(ENVIRONMENT_IS_PTHREAD){if(_pthread_self())__emscripten_thread_exit(EXITSTATUS);return}_exit(EXITSTATUS)}catch(e){handleException(e)}}};var callUserCallback=func=>{if(ABORT){return}try{func();maybeExit()}catch(e){handleException(e)}};function __emscripten_thread_mailbox_await(pthread_ptr){pthread_ptr=bigintToI53Checked(pthread_ptr);if(Atomics.waitAsync){var wait=Atomics.waitAsync((growMemViews(),HEAP32),pthread_ptr/4,pthread_ptr);wait.value.then(checkMailbox);var waitingAsync=pthread_ptr+228;Atomics.store((growMemViews(),HEAP32),waitingAsync/4,1)}}var checkMailbox=()=>callUserCallback(()=>{var pthread_ptr=_pthread_self();if(pthread_ptr){__emscripten_thread_mailbox_await(pthread_ptr);__emscripten_check_mailbox()}});function __emscripten_notify_mailbox_postmessage(targetThread,currThreadId){targetThread=bigintToI53Checked(targetThread);currThreadId=bigintToI53Checked(currThreadId);if(targetThread==currThreadId){setTimeout(checkMailbox)}else if(ENVIRONMENT_IS_PTHREAD){postMessage({targetThread,cmd:"checkMailbox"})}else{var worker=PThread.pthreads[targetThread];if(!worker){return}worker.postMessage({cmd:"checkMailbox"})}}var proxiedJSCallArgs=[];function __emscripten_receive_on_main_thread_js(funcIndex,emAsmAddr,callingThread,numCallArgs,args){emAsmAddr=bigintToI53Checked(emAsmAddr);callingThread=bigintToI53Checked(callingThread);args=bigintToI53Checked(args);numCallArgs/=2;proxiedJSCallArgs.length=numCallArgs;var b=args/8;for(var i=0;i<numCallArgs;i++){if((growMemViews(),HEAP64)[b+2*i]){proxiedJSCallArgs[i]=(growMemViews(),HEAP64)[b+2*i+1]}else{proxiedJSCallArgs[i]=(growMemViews(),HEAPF64)[b+2*i+1]}}var func=proxiedFunctionTable[funcIndex];PThread.currentProxiedOperationCallerThread=callingThread;var rtn=func(...proxiedJSCallArgs);PThread.currentProxiedOperationCallerThread=0;if(typeof rtn=="bigint"){rtn=bigintToI53Checked(rtn)}return rtn}function __emscripten_thread_cleanup(thread){thread=bigintToI53Checked(thread);if(!ENVIRONMENT_IS_PTHREAD)cleanupThread(thread);else postMessage({cmd:"cleanupThread",thread})}function __emscripten_thread_set_strongref(thread){thread=bigintToI53Checked(thread);if(ENVIRONMENT_IS_NODE){PThread.pthreads[thread].ref()}}var isLeapYear=year=>year%4===0&&(year%100!==0||year%400===0);var MONTH_DAYS_LEAP_CUMULATIVE=[0,31,60,91,121,152,182,213,244,274,305,335];var MONTH_DAYS_REGULAR_CUMULATIVE=[0,31,59,90,120,151,181,212,243,273,304,334];var ydayFromDate=date=>{var leap=isLeapYear(date.getFullYear());var monthDaysCumulative=leap?MONTH_DAYS_LEAP_CUMULATIVE:MONTH_DAYS_REGULAR_CUMULATIVE;var yday=monthDaysCumulative[date.getMonth()]+date.getDate()-1;return yday};function __localtime_js(time,tmPtr){time=bigintToI53Checked(time);tmPtr=bigintToI53Checked(tmPtr);var date=new Date(time*1e3);(growMemViews(),HEAP32)[tmPtr/4]=date.getSeconds();(growMemViews(),HEAP32)[(tmPtr+4)/4]=date.getMinutes();(growMemViews(),HEAP32)[(tmPtr+8)/4]=date.getHours();(growMemViews(),HEAP32)[(tmPtr+12)/4]=date.getDate();(growMemViews(),HEAP32)[(tmPtr+16)/4]=date.getMonth();(growMemViews(),HEAP32)[(tmPtr+20)/4]=date.getFullYear()-1900;(growMemViews(),HEAP32)[(tmPtr+24)/4]=date.getDay();var yday=ydayFromDate(date)|0;(growMemViews(),HEAP32)[(tmPtr+28)/4]=yday;(growMemViews(),HEAP64)[(tmPtr+40)/8]=BigInt(-(date.getTimezoneOffset()*60));var start=new Date(date.getFullYear(),0,1);var summerOffset=new Date(date.getFullYear(),6,1).getTimezoneOffset();var winterOffset=start.getTimezoneOffset();var dst=(summerOffset!=winterOffset&&date.getTimezoneOffset()==Math.min(winterOffset,summerOffset))|0;(growMemViews(),HEAP32)[(tmPtr+32)/4]=dst}function __mmap_js(len,prot,flags,fd,offset,allocated,addr){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(12,0,1,len,prot,flags,fd,offset,allocated,addr);len=bigintToI53Checked(len);offset=bigintToI53Checked(offset);allocated=bigintToI53Checked(allocated);addr=bigintToI53Checked(addr);try{var stream=SYSCALLS.getStreamFromFD(fd);var res=FS.mmap(stream,len,offset,prot,flags);var ptr=res.ptr;(growMemViews(),HEAP32)[allocated/4]=res.allocated;(growMemViews(),HEAPU64)[addr/8]=BigInt(ptr);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function __munmap_js(addr,len,prot,flags,fd,offset){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(13,0,1,addr,len,prot,flags,fd,offset);addr=bigintToI53Checked(addr);len=bigintToI53Checked(len);offset=bigintToI53Checked(offset);try{var stream=SYSCALLS.getStreamFromFD(fd);if(prot&2){SYSCALLS.doMsync(addr,stream,len,flags,offset)}}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var __tzset_js=function(timezone,daylight,std_name,dst_name){timezone=bigintToI53Checked(timezone);daylight=bigintToI53Checked(daylight);std_name=bigintToI53Checked(std_name);dst_name=bigintToI53Checked(dst_name);var currentYear=(new Date).getFullYear();var winter=new Date(currentYear,0,1);var summer=new Date(currentYear,6,1);var winterOffset=winter.getTimezoneOffset();var summerOffset=summer.getTimezoneOffset();var stdTimezoneOffset=Math.max(winterOffset,summerOffset);(growMemViews(),HEAPU64)[timezone/8]=BigInt(stdTimezoneOffset*60);(growMemViews(),HEAP32)[daylight/4]=Number(winterOffset!=summerOffset);var extractZone=timezoneOffset=>{var sign=timezoneOffset>=0?"-":"+";var absOffset=Math.abs(timezoneOffset);var hours=String(Math.floor(absOffset/60)).padStart(2,"0");var minutes=String(absOffset%60).padStart(2,"0");return`UTC${sign}${hours}${minutes}`};var winterName=extractZone(winterOffset);var summerName=extractZone(summerOffset);if(summerOffset<winterOffset){stringToUTF8(winterName,std_name,17);stringToUTF8(summerName,dst_name,17)}else{stringToUTF8(winterName,dst_name,17);stringToUTF8(summerName,std_name,17)}};var _emscripten_get_now=()=>performance.timeOrigin+performance.now();var _emscripten_date_now=()=>Date.now();var nowIsMonotonic=1;var checkWasiClock=clock_id=>clock_id>=0&&clock_id<=3;function _clock_time_get(clk_id,ignored_precision,ptime){ignored_precision=bigintToI53Checked(ignored_precision);ptime=bigintToI53Checked(ptime);if(!checkWasiClock(clk_id)){return 28}var now;if(clk_id===0){now=_emscripten_date_now()}else if(nowIsMonotonic){now=_emscripten_get_now()}else{return 52}var nsec=Math.round(now*1e3*1e3);(growMemViews(),HEAP64)[ptime/8]=BigInt(nsec);return 0}var _emscripten_check_blocking_allowed=()=>{};var runtimeKeepalivePush=()=>{runtimeKeepaliveCounter+=1};var _emscripten_exit_with_live_runtime=()=>{runtimeKeepalivePush();throw"unwind"};var jsStackTrace=()=>(new Error).stack.toString();var getCallstack=flags=>{var callstack=jsStackTrace();var lines=callstack.split("\\n");callstack="";var firefoxRe=new RegExp("\\\\s*(.*?)@(.*?):([0-9]+):([0-9]+)");var chromeRe=new RegExp("\\\\s*at (.*?) \\\\((.*):(.*):(.*)\\\\)");for(var line of lines){var symbolName="";var file="";var lineno=0;var column=0;var parts=chromeRe.exec(line);if(parts?.length==5){symbolName=parts[1];file=parts[2];lineno=parts[3];column=parts[4]}else{parts=firefoxRe.exec(line);if(parts?.length>=4){symbolName=parts[1];file=parts[2];lineno=parts[3];column=parts[4]|0}else{callstack+=line+"\\n";continue}}if(symbolName=="_emscripten_log"||symbolName=="_emscripten_get_callstack"){callstack="";continue}if(flags&24){if(flags&64){file=file.substring(file.replace(/\\\\/g,"/").lastIndexOf("/")+1)}callstack+=`    at ${symbolName} (${file}:${lineno}:${column})\\n`}}callstack=callstack.replace(/\\s+$/,"");return callstack};function _emscripten_get_callstack(flags,str,maxbytes){str=bigintToI53Checked(str);var callstack=getCallstack(flags);if(!str||maxbytes<=0){return lengthBytesUTF8(callstack)+1}var bytesWrittenExcludingNull=stringToUTF8(callstack,str,maxbytes);return bytesWrittenExcludingNull+1}var getHeapMax=()=>4294967296;var _emscripten_get_heap_max=()=>BigInt(getHeapMax());var _emscripten_has_asyncify=()=>2;var _emscripten_num_logical_cores=()=>ENVIRONMENT_IS_NODE?require("os").cpus().length:navigator["hardwareConcurrency"];var growMemory=size=>{var oldHeapSize=wasmMemory.buffer.byteLength;var pages=(size-oldHeapSize+65535)/65536|0;try{wasmMemory.grow(BigInt(pages));updateMemoryViews();return 1}catch(e){}};function _emscripten_resize_heap(requestedSize){requestedSize=bigintToI53Checked(requestedSize);var oldSize=(growMemViews(),HEAPU8).length;if(requestedSize<=oldSize){return false}var maxHeapSize=getHeapMax();if(requestedSize>maxHeapSize){return false}for(var cutDown=1;cutDown<=4;cutDown*=2){var overGrownHeapSize=oldSize*(1+.2/cutDown);overGrownHeapSize=Math.min(overGrownHeapSize,requestedSize+100663296);var newSize=Math.min(maxHeapSize,alignMemory(Math.max(requestedSize,overGrownHeapSize),65536));var replacement=growMemory(newSize);if(replacement){return true}}return false}var stringToUTF8OnStack=str=>{var size=lengthBytesUTF8(str)+1;var ret=stackAlloc(size);stringToUTF8(str,ret,size);return ret};var writeI53ToI64=(ptr,num)=>{(growMemViews(),HEAPU32)[ptr/4]=num;var lower=(growMemViews(),HEAPU32)[ptr/4];(growMemViews(),HEAPU32)[(ptr+4)/4]=(num-lower)/4294967296};var stringToNewUTF8=str=>{var size=lengthBytesUTF8(str)+1;var ret=_malloc(size);if(ret)stringToUTF8(str,ret,size);return ret};var readI53FromI64=ptr=>(growMemViews(),HEAPU32)[ptr/4]+(growMemViews(),HEAP32)[(ptr+4)/4]*4294967296;var WebGPU={Internals:{jsObjects:[],jsObjectInsert:(ptr,jsObject)=>{WebGPU.Internals.jsObjects[ptr]=jsObject},bufferOnUnmaps:[],futures:[],futureInsert:(futureId,promise)=>{WebGPU.Internals.futures[futureId]=new Promise(resolve=>promise.finally(()=>resolve(futureId)))}},getJsObject:ptr=>{if(!ptr)return undefined;return WebGPU.Internals.jsObjects[ptr]},importJsAdapter:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateAdapter(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBindGroup:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateBindGroup(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBindGroupLayout:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateBindGroupLayout(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBuffer:(buffer,parentPtr=0)=>{assert(buffer.mapState==="unmapped");var bufferPtr=_emwgpuCreateBuffer(parentPtr);WebGPU.Internals.jsObjectInsert(bufferPtr,buffer);return bufferPtr},importJsCommandBuffer:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateCommandBuffer(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsCommandEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateCommandEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsComputePassEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateComputePassEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsComputePipeline:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateComputePipeline(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsDevice:(device,parentPtr=0)=>{var queuePtr=_emwgpuCreateQueue(parentPtr);var devicePtr=_emwgpuCreateDevice(parentPtr,queuePtr);WebGPU.Internals.jsObjectInsert(queuePtr,device.queue);WebGPU.Internals.jsObjectInsert(devicePtr,device);return devicePtr},importJsExternalTexture:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateExternalTexture(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsPipelineLayout:(obj,parentPtr=0)=>{var ptr=_emwgpuCreatePipelineLayout(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsQuerySet:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateQuerySet(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsQueue:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateQueue(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderBundle:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderBundle(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderBundleEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderBundleEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderPassEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderPassEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderPipeline:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderPipeline(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsSampler:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateSampler(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsShaderModule:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateShaderModule(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsSurface:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateSurface(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsTexture:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateTexture(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsTextureView:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateTextureView(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},errorCallback:(callback,type,message,userdata)=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(message);((a1,a2,a3)=>getWasmTableEntry(callback).call(null,a1,BigInt(a2),BigInt(a3)))(type,BigInt(messagePtr),userdata);stackRestore(sp)},iterateExtensions:(root,handlers)=>{for(var ptr=Number((growMemViews(),HEAPU64)[root/8]);ptr;ptr=Number((growMemViews(),HEAPU64)[ptr/8])){var sType=(growMemViews(),HEAP32)[(ptr+8)/4];var handler=handlers[sType](ptr)}},setStringView:(ptr,data,length)=>{(growMemViews(),HEAPU64)[ptr/8]=BigInt(data);(growMemViews(),HEAPU64)[(ptr+8)/8]=BigInt(length)},makeStringFromStringView:stringViewPtr=>{var ptr=Number((growMemViews(),HEAPU64)[stringViewPtr/8]);var length=Number((growMemViews(),HEAPU64)[(stringViewPtr+8)/8]);return UTF8ToString(ptr,length)},makeStringFromOptionalStringView:stringViewPtr=>{var ptr=Number((growMemViews(),HEAPU64)[stringViewPtr/8]);var length=Number((growMemViews(),HEAPU64)[(stringViewPtr+8)/8]);if(!ptr){if(length===0){return""}return undefined}return UTF8ToString(ptr,length)},makeColor:ptr=>({r:(growMemViews(),HEAPF64)[ptr/8],g:(growMemViews(),HEAPF64)[(ptr+8)/8],b:(growMemViews(),HEAPF64)[(ptr+16)/8],a:(growMemViews(),HEAPF64)[(ptr+24)/8]}),makeExtent3D:ptr=>({width:(growMemViews(),HEAPU32)[ptr/4],height:(growMemViews(),HEAPU32)[(ptr+4)/4],depthOrArrayLayers:(growMemViews(),HEAPU32)[(ptr+8)/4]}),makeOrigin3D:ptr=>({x:(growMemViews(),HEAPU32)[ptr/4],y:(growMemViews(),HEAPU32)[(ptr+4)/4],z:(growMemViews(),HEAPU32)[(ptr+8)/4]}),makeTexelCopyTextureInfo:ptr=>({texture:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[ptr/8])),mipLevel:(growMemViews(),HEAPU32)[(ptr+8)/4],origin:WebGPU.makeOrigin3D(ptr+12),aspect:WebGPU.TextureAspect[(growMemViews(),HEAP32)[(ptr+24)/4]]}),makeTexelCopyBufferLayout:ptr=>{var bytesPerRow=(growMemViews(),HEAPU32)[(ptr+8)/4];var rowsPerImage=(growMemViews(),HEAPU32)[(ptr+12)/4];return{offset:readI53FromI64(ptr),bytesPerRow:bytesPerRow===4294967295?undefined:bytesPerRow,rowsPerImage:rowsPerImage===4294967295?undefined:rowsPerImage}},makeTexelCopyBufferInfo:ptr=>{var layoutPtr=ptr+0;var bufferCopyView=WebGPU.makeTexelCopyBufferLayout(layoutPtr);bufferCopyView["buffer"]=WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+16)/8]));return bufferCopyView},makePassTimestampWrites:ptr=>{if(ptr===0)return undefined;return{querySet:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+8)/8])),beginningOfPassWriteIndex:(growMemViews(),HEAPU32)[(ptr+16)/4],endOfPassWriteIndex:(growMemViews(),HEAPU32)[(ptr+20)/4]}},makePipelineConstants:(constantCount,constantsPtr)=>{if(!constantCount)return;var constants={};for(var i=0;i<constantCount;++i){var entryPtr=constantsPtr+32*i;var key=WebGPU.makeStringFromStringView(entryPtr+8);constants[key]=(growMemViews(),HEAPF64)[(entryPtr+24)/8]}return constants},makePipelineLayout:layoutPtr=>{if(!layoutPtr)return"auto";return WebGPU.getJsObject(layoutPtr)},makeComputeState:ptr=>{if(!ptr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(ptr+32)/8]),Number((growMemViews(),HEAPU64)[(ptr+40)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(ptr+16)};return desc},makeComputePipelineDesc:descriptor=>{var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.makePipelineLayout(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),compute:WebGPU.makeComputeState(descriptor+32)};return desc},makeRenderPipelineDesc:descriptor=>{function makePrimitiveState(psPtr){if(!psPtr)return undefined;return{topology:WebGPU.PrimitiveTopology[(growMemViews(),HEAP32)[(psPtr+8)/4]],stripIndexFormat:WebGPU.IndexFormat[(growMemViews(),HEAP32)[(psPtr+12)/4]],frontFace:WebGPU.FrontFace[(growMemViews(),HEAP32)[(psPtr+16)/4]],cullMode:WebGPU.CullMode[(growMemViews(),HEAP32)[(psPtr+20)/4]],unclippedDepth:!!(growMemViews(),HEAPU32)[(psPtr+24)/4]}}function makeBlendComponent(bdPtr){if(!bdPtr)return undefined;return{operation:WebGPU.BlendOperation[(growMemViews(),HEAP32)[bdPtr/4]],srcFactor:WebGPU.BlendFactor[(growMemViews(),HEAP32)[(bdPtr+4)/4]],dstFactor:WebGPU.BlendFactor[(growMemViews(),HEAP32)[(bdPtr+8)/4]]}}function makeBlendState(bsPtr){if(!bsPtr)return undefined;return{alpha:makeBlendComponent(bsPtr+12),color:makeBlendComponent(bsPtr+0)}}function makeColorState(csPtr){var format=WebGPU.TextureFormat[(growMemViews(),HEAP32)[(csPtr+8)/4]];return format?{format,blend:makeBlendState(Number((growMemViews(),HEAPU64)[(csPtr+16)/8])),writeMask:(growMemViews(),HEAPU32)[(csPtr+24)/4]}:undefined}function makeColorStates(count,csArrayPtr){var states=[];for(var i=0;i<count;++i){states.push(makeColorState(csArrayPtr+32*i))}return states}function makeStencilStateFace(ssfPtr){return{compare:WebGPU.CompareFunction[(growMemViews(),HEAP32)[ssfPtr/4]],failOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+4)/4]],depthFailOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+8)/4]],passOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+12)/4]]}}function makeDepthStencilState(dssPtr){if(!dssPtr)return undefined;return{format:WebGPU.TextureFormat[(growMemViews(),HEAP32)[(dssPtr+8)/4]],depthWriteEnabled:!!(growMemViews(),HEAPU32)[(dssPtr+12)/4],depthCompare:WebGPU.CompareFunction[(growMemViews(),HEAP32)[(dssPtr+16)/4]],stencilFront:makeStencilStateFace(dssPtr+20),stencilBack:makeStencilStateFace(dssPtr+36),stencilReadMask:(growMemViews(),HEAPU32)[(dssPtr+52)/4],stencilWriteMask:(growMemViews(),HEAPU32)[(dssPtr+56)/4],depthBias:(growMemViews(),HEAP32)[(dssPtr+60)/4],depthBiasSlopeScale:(growMemViews(),HEAPF32)[(dssPtr+64)/4],depthBiasClamp:(growMemViews(),HEAPF32)[(dssPtr+68)/4]}}function makeVertexAttribute(vaPtr){return{format:WebGPU.VertexFormat[(growMemViews(),HEAP32)[(vaPtr+8)/4]],offset:readI53FromI64(vaPtr+16),shaderLocation:(growMemViews(),HEAPU32)[(vaPtr+24)/4]}}function makeVertexAttributes(count,vaArrayPtr){var vas=[];for(var i=0;i<count;++i){vas.push(makeVertexAttribute(vaArrayPtr+i*32))}return vas}function makeVertexBuffer(vbPtr){if(!vbPtr)return undefined;var stepMode=WebGPU.VertexStepMode[(growMemViews(),HEAP32)[(vbPtr+8)/4]];var attributeCount=Number((growMemViews(),HEAPU64)[(vbPtr+24)/8]);if(!stepMode&&!attributeCount){return null}return{arrayStride:readI53FromI64(vbPtr+16),stepMode,attributes:makeVertexAttributes(attributeCount,Number((growMemViews(),HEAPU64)[(vbPtr+32)/8]))}}function makeVertexBuffers(count,vbArrayPtr){if(!count)return undefined;var vbs=[];for(var i=0;i<count;++i){vbs.push(makeVertexBuffer(vbArrayPtr+i*40))}return vbs}function makeVertexState(viPtr){if(!viPtr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(viPtr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(viPtr+32)/8]),Number((growMemViews(),HEAPU64)[(viPtr+40)/8])),buffers:makeVertexBuffers(Number((growMemViews(),HEAPU64)[(viPtr+48)/8]),Number((growMemViews(),HEAPU64)[(viPtr+56)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(viPtr+16)};return desc}function makeMultisampleState(msPtr){if(!msPtr)return undefined;return{count:(growMemViews(),HEAPU32)[(msPtr+8)/4],mask:(growMemViews(),HEAPU32)[(msPtr+12)/4],alphaToCoverageEnabled:!!(growMemViews(),HEAPU32)[(msPtr+16)/4]}}function makeFragmentState(fsPtr){if(!fsPtr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(fsPtr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(fsPtr+32)/8]),Number((growMemViews(),HEAPU64)[(fsPtr+40)/8])),targets:makeColorStates(Number((growMemViews(),HEAPU64)[(fsPtr+48)/8]),Number((growMemViews(),HEAPU64)[(fsPtr+56)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(fsPtr+16)};return desc}var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.makePipelineLayout(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),vertex:makeVertexState(descriptor+32),primitive:makePrimitiveState(descriptor+96),depthStencil:makeDepthStencilState(Number((growMemViews(),HEAPU64)[(descriptor+128)/8])),multisample:makeMultisampleState(descriptor+136),fragment:makeFragmentState(Number((growMemViews(),HEAPU64)[(descriptor+160)/8]))};return desc},fillLimitStruct:(limits,limitsOutPtr)=>{var nextInChainPtr=Number((growMemViews(),HEAPU64)[limitsOutPtr/8]);function setLimitValueU32(name,basePtr,limitOffset,fallbackValue=0){var limitValue=limits[name]??fallbackValue;(growMemViews(),HEAPU32)[(basePtr+limitOffset)/4]=limitValue}function setLimitValueU64(name,basePtr,limitOffset,fallbackValue=0){var limitValue=limits[name]??fallbackValue;writeI53ToI64(basePtr+limitOffset,limitValue)}setLimitValueU32("maxTextureDimension1D",limitsOutPtr,8);setLimitValueU32("maxTextureDimension2D",limitsOutPtr,12);setLimitValueU32("maxTextureDimension3D",limitsOutPtr,16);setLimitValueU32("maxTextureArrayLayers",limitsOutPtr,20);setLimitValueU32("maxBindGroups",limitsOutPtr,24);setLimitValueU32("maxBindGroupsPlusVertexBuffers",limitsOutPtr,28);setLimitValueU32("maxBindingsPerBindGroup",limitsOutPtr,32);setLimitValueU32("maxDynamicUniformBuffersPerPipelineLayout",limitsOutPtr,36);setLimitValueU32("maxDynamicStorageBuffersPerPipelineLayout",limitsOutPtr,40);setLimitValueU32("maxSampledTexturesPerShaderStage",limitsOutPtr,44);setLimitValueU32("maxSamplersPerShaderStage",limitsOutPtr,48);setLimitValueU32("maxStorageBuffersPerShaderStage",limitsOutPtr,52);setLimitValueU32("maxStorageTexturesPerShaderStage",limitsOutPtr,56);setLimitValueU32("maxUniformBuffersPerShaderStage",limitsOutPtr,60);setLimitValueU32("minUniformBufferOffsetAlignment",limitsOutPtr,80);setLimitValueU32("minStorageBufferOffsetAlignment",limitsOutPtr,84);setLimitValueU64("maxUniformBufferBindingSize",limitsOutPtr,64);setLimitValueU64("maxStorageBufferBindingSize",limitsOutPtr,72);setLimitValueU32("maxVertexBuffers",limitsOutPtr,88);setLimitValueU64("maxBufferSize",limitsOutPtr,96);setLimitValueU32("maxVertexAttributes",limitsOutPtr,104);setLimitValueU32("maxVertexBufferArrayStride",limitsOutPtr,108);setLimitValueU32("maxInterStageShaderVariables",limitsOutPtr,112);setLimitValueU32("maxColorAttachments",limitsOutPtr,116);setLimitValueU32("maxColorAttachmentBytesPerSample",limitsOutPtr,120);setLimitValueU32("maxComputeWorkgroupStorageSize",limitsOutPtr,124);setLimitValueU32("maxComputeInvocationsPerWorkgroup",limitsOutPtr,128);setLimitValueU32("maxComputeWorkgroupSizeX",limitsOutPtr,132);setLimitValueU32("maxComputeWorkgroupSizeY",limitsOutPtr,136);setLimitValueU32("maxComputeWorkgroupSizeZ",limitsOutPtr,140);setLimitValueU32("maxComputeWorkgroupsPerDimension",limitsOutPtr,144);setLimitValueU32("maxImmediateSize",limitsOutPtr,148);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var compatibilityModeLimitsPtr=nextInChainPtr;setLimitValueU32("maxStorageBuffersInVertexStage",compatibilityModeLimitsPtr,16,limits.maxStorageBuffersPerShaderStage);setLimitValueU32("maxStorageBuffersInFragmentStage",compatibilityModeLimitsPtr,24,limits.maxStorageBuffersPerShaderStage);setLimitValueU32("maxStorageTexturesInVertexStage",compatibilityModeLimitsPtr,20,limits.maxStorageTexturesPerShaderStage);setLimitValueU32("maxStorageTexturesInFragmentStage",compatibilityModeLimitsPtr,28,limits.maxStorageTexturesPerShaderStage)}},fillAdapterInfoStruct:(info,infoStruct)=>{(growMemViews(),HEAPU32)[(infoStruct+88)/4]=info.subgroupMinSize;(growMemViews(),HEAPU32)[(infoStruct+92)/4]=info.subgroupMaxSize;var strs=info.vendor+info.architecture+info.device+info.description;var strPtr=stringToNewUTF8(strs);var vendorLen=lengthBytesUTF8(info.vendor);WebGPU.setStringView(infoStruct+8,strPtr,vendorLen);strPtr+=vendorLen;var architectureLen=lengthBytesUTF8(info.architecture);WebGPU.setStringView(infoStruct+24,strPtr,architectureLen);strPtr+=architectureLen;var deviceLen=lengthBytesUTF8(info.device);WebGPU.setStringView(infoStruct+40,strPtr,deviceLen);strPtr+=deviceLen;var descriptionLen=lengthBytesUTF8(info.description);WebGPU.setStringView(infoStruct+56,strPtr,descriptionLen);strPtr+=descriptionLen;(growMemViews(),HEAP32)[(infoStruct+72)/4]=2;var adapterType=info.isFallbackAdapter?3:4;(growMemViews(),HEAP32)[(infoStruct+76)/4]=adapterType;(growMemViews(),HEAPU32)[(infoStruct+80)/4]=0;(growMemViews(),HEAPU32)[(infoStruct+84)/4]=0},AddressMode:[,"clamp-to-edge","repeat","mirror-repeat"],BlendFactor:[,"zero","one","src","one-minus-src","src-alpha","one-minus-src-alpha","dst","one-minus-dst","dst-alpha","one-minus-dst-alpha","src-alpha-saturated","constant","one-minus-constant","src1","one-minus-src1","src1-alpha","one-minus-src1-alpha"],BlendOperation:[,"add","subtract","reverse-subtract","min","max"],BufferBindingType:[,,"uniform","storage","read-only-storage"],BufferMapState:[,"unmapped","pending","mapped"],CompareFunction:[,"never","less","equal","less-equal","greater","not-equal","greater-equal","always"],CompilationInfoRequestStatus:[,"success","callback-cancelled"],ComponentSwizzle:[,"0","1","r","g","b","a"],CompositeAlphaMode:[,"opaque","premultiplied","unpremultiplied","inherit"],CullMode:[,"none","front","back"],ErrorFilter:[,"validation","out-of-memory","internal"],FeatureLevel:[,"compatibility","core"],FeatureName:{1:"core-features-and-limits",2:"depth-clip-control",3:"depth32float-stencil8",4:"texture-compression-bc",5:"texture-compression-bc-sliced-3d",6:"texture-compression-etc2",7:"texture-compression-astc",8:"texture-compression-astc-sliced-3d",9:"timestamp-query",10:"indirect-first-instance",11:"shader-f16",12:"rg11b10ufloat-renderable",13:"bgra8unorm-storage",14:"float32-filterable",15:"float32-blendable",16:"clip-distances",17:"dual-source-blending",18:"subgroups",19:"texture-formats-tier1",20:"texture-formats-tier2",21:"primitive-index",22:"texture-component-swizzle",327692:"chromium-experimental-unorm16-texture-formats",327729:"chromium-experimental-multi-draw-indirect"},FilterMode:[,"nearest","linear"],FrontFace:[,"ccw","cw"],IndexFormat:[,"uint16","uint32"],InstanceFeatureName:[,"timed-wait-any","shader-source-spirv","multiple-devices-per-adapter"],LoadOp:[,"load","clear"],MipmapFilterMode:[,"nearest","linear"],OptionalBool:["false","true"],PowerPreference:[,"low-power","high-performance"],PredefinedColorSpace:[,"srgb","display-p3"],PrimitiveTopology:[,"point-list","line-list","line-strip","triangle-list","triangle-strip"],QueryType:[,"occlusion","timestamp"],SamplerBindingType:[,,"filtering","non-filtering","comparison"],Status:[,"success","error"],StencilOperation:[,"keep","zero","replace","invert","increment-clamp","decrement-clamp","increment-wrap","decrement-wrap"],StorageTextureAccess:[,,"write-only","read-only","read-write"],StoreOp:[,"store","discard"],SurfaceGetCurrentTextureStatus:[,"success-optimal","success-suboptimal","timeout","outdated","lost","error"],TextureAspect:[,"all","stencil-only","depth-only"],TextureDimension:[,"1d","2d","3d"],TextureFormat:[,"r8unorm","r8snorm","r8uint","r8sint","r16unorm","r16snorm","r16uint","r16sint","r16float","rg8unorm","rg8snorm","rg8uint","rg8sint","r32float","r32uint","r32sint","rg16unorm","rg16snorm","rg16uint","rg16sint","rg16float","rgba8unorm","rgba8unorm-srgb","rgba8snorm","rgba8uint","rgba8sint","bgra8unorm","bgra8unorm-srgb","rgb10a2uint","rgb10a2unorm","rg11b10ufloat","rgb9e5ufloat","rg32float","rg32uint","rg32sint","rgba16unorm","rgba16snorm","rgba16uint","rgba16sint","rgba16float","rgba32float","rgba32uint","rgba32sint","stencil8","depth16unorm","depth24plus","depth24plus-stencil8","depth32float","depth32float-stencil8","bc1-rgba-unorm","bc1-rgba-unorm-srgb","bc2-rgba-unorm","bc2-rgba-unorm-srgb","bc3-rgba-unorm","bc3-rgba-unorm-srgb","bc4-r-unorm","bc4-r-snorm","bc5-rg-unorm","bc5-rg-snorm","bc6h-rgb-ufloat","bc6h-rgb-float","bc7-rgba-unorm","bc7-rgba-unorm-srgb","etc2-rgb8unorm","etc2-rgb8unorm-srgb","etc2-rgb8a1unorm","etc2-rgb8a1unorm-srgb","etc2-rgba8unorm","etc2-rgba8unorm-srgb","eac-r11unorm","eac-r11snorm","eac-rg11unorm","eac-rg11snorm","astc-4x4-unorm","astc-4x4-unorm-srgb","astc-5x4-unorm","astc-5x4-unorm-srgb","astc-5x5-unorm","astc-5x5-unorm-srgb","astc-6x5-unorm","astc-6x5-unorm-srgb","astc-6x6-unorm","astc-6x6-unorm-srgb","astc-8x5-unorm","astc-8x5-unorm-srgb","astc-8x6-unorm","astc-8x6-unorm-srgb","astc-8x8-unorm","astc-8x8-unorm-srgb","astc-10x5-unorm","astc-10x5-unorm-srgb","astc-10x6-unorm","astc-10x6-unorm-srgb","astc-10x8-unorm","astc-10x8-unorm-srgb","astc-10x10-unorm","astc-10x10-unorm-srgb","astc-12x10-unorm","astc-12x10-unorm-srgb","astc-12x12-unorm","astc-12x12-unorm-srgb"],TextureSampleType:[,,"float","unfilterable-float","depth","sint","uint"],TextureViewDimension:[,"1d","2d","2d-array","cube","cube-array","3d"],ToneMappingMode:[,"standard","extended"],VertexFormat:[,"uint8","uint8x2","uint8x4","sint8","sint8x2","sint8x4","unorm8","unorm8x2","unorm8x4","snorm8","snorm8x2","snorm8x4","uint16","uint16x2","uint16x4","sint16","sint16x2","sint16x4","unorm16","unorm16x2","unorm16x4","snorm16","snorm16x2","snorm16x4","float16","float16x2","float16x4","float32","float32x2","float32x3","float32x4","uint32","uint32x2","uint32x3","uint32x4","sint32","sint32x2","sint32x3","sint32x4","unorm10-10-10-2","unorm8x4-bgra"],VertexStepMode:[,"vertex","instance"],WGSLLanguageFeatureName:[,"readonly_and_readwrite_storage_textures","packed_4x8_integer_dot_product","unrestricted_pointer_parameters","pointer_composite_access","uniform_buffer_standard_layout","subgroup_id","texture_and_sampler_let","subgroup_uniformity","texture_formats_tier1"]};var emwgpuStringToInt_DeviceLostReason={undefined:1,unknown:1,destroyed:2};var runtimeKeepalivePop=()=>{runtimeKeepaliveCounter-=1};function _emwgpuAdapterRequestDevice(adapterPtr,futureId,deviceLostFutureId,devicePtr,queuePtr,descriptor){adapterPtr=bigintToI53Checked(adapterPtr);futureId=bigintToI53Checked(futureId);deviceLostFutureId=bigintToI53Checked(deviceLostFutureId);devicePtr=bigintToI53Checked(devicePtr);queuePtr=bigintToI53Checked(queuePtr);descriptor=bigintToI53Checked(descriptor);var adapter=WebGPU.getJsObject(adapterPtr);var desc={};if(descriptor){var requiredFeatureCount=Number((growMemViews(),HEAPU64)[(descriptor+24)/8]);if(requiredFeatureCount){var requiredFeaturesPtr=Number((growMemViews(),HEAPU64)[(descriptor+32)/8]);desc["requiredFeatures"]=Array.from((growMemViews(),HEAPU32).subarray(requiredFeaturesPtr/4,(requiredFeaturesPtr+requiredFeatureCount*4)/4),feature=>WebGPU.FeatureName[feature])}var limitsPtr=Number((growMemViews(),HEAPU64)[(descriptor+40)/8]);if(limitsPtr){var nextInChainPtr=Number((growMemViews(),HEAPU64)[limitsPtr/8]);var requiredLimits={};function setLimitU32IfDefined(name,basePtr,limitOffset,ignoreIfZero=false){var ptr=basePtr+limitOffset;var value=(growMemViews(),HEAPU32)[ptr/4];if(value!=4294967295&&(!ignoreIfZero||value!=0)){requiredLimits[name]=value}}function setLimitU64IfDefined(name,basePtr,limitOffset){var ptr=basePtr+limitOffset;var limitPart1=(growMemViews(),HEAPU32)[ptr/4];var limitPart2=(growMemViews(),HEAPU32)[(ptr+4)/4];if(limitPart1!=4294967295||limitPart2!=4294967295){requiredLimits[name]=readI53FromI64(ptr)}}setLimitU32IfDefined("maxTextureDimension1D",limitsPtr,8);setLimitU32IfDefined("maxTextureDimension2D",limitsPtr,12);setLimitU32IfDefined("maxTextureDimension3D",limitsPtr,16);setLimitU32IfDefined("maxTextureArrayLayers",limitsPtr,20);setLimitU32IfDefined("maxBindGroups",limitsPtr,24);setLimitU32IfDefined("maxBindGroupsPlusVertexBuffers",limitsPtr,28);setLimitU32IfDefined("maxBindingsPerBindGroup",limitsPtr,32);setLimitU32IfDefined("maxDynamicUniformBuffersPerPipelineLayout",limitsPtr,36);setLimitU32IfDefined("maxDynamicStorageBuffersPerPipelineLayout",limitsPtr,40);setLimitU32IfDefined("maxSampledTexturesPerShaderStage",limitsPtr,44);setLimitU32IfDefined("maxSamplersPerShaderStage",limitsPtr,48);setLimitU32IfDefined("maxStorageBuffersPerShaderStage",limitsPtr,52);setLimitU32IfDefined("maxStorageTexturesPerShaderStage",limitsPtr,56);setLimitU32IfDefined("maxUniformBuffersPerShaderStage",limitsPtr,60);setLimitU32IfDefined("minUniformBufferOffsetAlignment",limitsPtr,80);setLimitU32IfDefined("minStorageBufferOffsetAlignment",limitsPtr,84);setLimitU64IfDefined("maxUniformBufferBindingSize",limitsPtr,64);setLimitU64IfDefined("maxStorageBufferBindingSize",limitsPtr,72);setLimitU32IfDefined("maxVertexBuffers",limitsPtr,88);setLimitU64IfDefined("maxBufferSize",limitsPtr,96);setLimitU32IfDefined("maxVertexAttributes",limitsPtr,104);setLimitU32IfDefined("maxVertexBufferArrayStride",limitsPtr,108);setLimitU32IfDefined("maxInterStageShaderVariables",limitsPtr,112);setLimitU32IfDefined("maxColorAttachments",limitsPtr,116);setLimitU32IfDefined("maxColorAttachmentBytesPerSample",limitsPtr,120);setLimitU32IfDefined("maxComputeWorkgroupStorageSize",limitsPtr,124);setLimitU32IfDefined("maxComputeInvocationsPerWorkgroup",limitsPtr,128);setLimitU32IfDefined("maxComputeWorkgroupSizeX",limitsPtr,132);setLimitU32IfDefined("maxComputeWorkgroupSizeY",limitsPtr,136);setLimitU32IfDefined("maxComputeWorkgroupSizeZ",limitsPtr,140);setLimitU32IfDefined("maxComputeWorkgroupsPerDimension",limitsPtr,144);setLimitU32IfDefined("maxImmediateSize",limitsPtr,148,true);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var compatibilityModeLimitsPtr=nextInChainPtr;if("maxStorageBuffersInVertexStage"in GPUSupportedLimits.prototype){setLimitU32IfDefined("maxStorageBuffersInVertexStage",compatibilityModeLimitsPtr,16);setLimitU32IfDefined("maxStorageTexturesInVertexStage",compatibilityModeLimitsPtr,20);setLimitU32IfDefined("maxStorageBuffersInFragmentStage",compatibilityModeLimitsPtr,24);setLimitU32IfDefined("maxStorageTexturesInFragmentStage",compatibilityModeLimitsPtr,28)}}desc["requiredLimits"]=requiredLimits}var defaultQueuePtr=Number((growMemViews(),HEAPU64)[(descriptor+48)/8]);if(defaultQueuePtr){var defaultQueueDesc={label:WebGPU.makeStringFromOptionalStringView(defaultQueuePtr+8)};desc["defaultQueue"]=defaultQueueDesc}desc["label"]=WebGPU.makeStringFromOptionalStringView(descriptor+8)}runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,adapter.requestDevice(desc).then(device=>{runtimeKeepalivePop();callUserCallback(()=>{WebGPU.Internals.jsObjectInsert(queuePtr,device.queue);WebGPU.Internals.jsObjectInsert(devicePtr,device);devicePtr=BigInt(devicePtr);WebGPU.Internals.futureInsert(deviceLostFutureId,device.lost.then(info=>{callUserCallback(()=>{device.onuncapturederror=ev=>{};var sp=stackSave();var messagePtr=stringToUTF8OnStack(info.message);_emwgpuOnDeviceLostCompleted(deviceLostFutureId,emwgpuStringToInt_DeviceLostReason[info.reason],BigInt(messagePtr));stackRestore(sp)})}));device.onuncapturederror=ev=>{var type=5;if(ev.error instanceof GPUValidationError)type=2;else if(ev.error instanceof GPUOutOfMemoryError)type=3;else if(ev.error instanceof GPUInternalError)type=4;var sp=stackSave();var messagePtr=stringToUTF8OnStack(ev.error.message);_emwgpuOnUncapturedError(BigInt(devicePtr),type,BigInt(messagePtr));stackRestore(sp)};_emwgpuOnRequestDeviceCompleted(futureId,1,BigInt(devicePtr),0n)})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);_emwgpuOnRequestDeviceCompleted(futureId,3,BigInt(devicePtr),BigInt(messagePtr));if(deviceLostFutureId){_emwgpuOnDeviceLostCompleted(deviceLostFutureId,4,BigInt(messagePtr))}stackRestore(sp)})}))}function _emwgpuBufferDestroy(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var buffer=WebGPU.getJsObject(bufferPtr);var onUnmap=WebGPU.Internals.bufferOnUnmaps[bufferPtr];if(onUnmap){for(var i=0;i<onUnmap.length;++i){onUnmap[i]()}delete WebGPU.Internals.bufferOnUnmaps[bufferPtr]}buffer.destroy()}var warnOnce=text=>{warnOnce.shown||={};if(!warnOnce.shown[text]){warnOnce.shown[text]=1;if(ENVIRONMENT_IS_NODE)text="warning: "+text;err(text)}};var _emwgpuBufferGetConstMappedRange=function(bufferPtr,offset,size){bufferPtr=bigintToI53Checked(bufferPtr);offset=bigintToI53Checked(offset);size=bigintToI53Checked(size);var ret=(()=>{var buffer=WebGPU.getJsObject(bufferPtr);if(size==-1)size=undefined;var mapped;try{mapped=buffer.getMappedRange(offset,size)}catch(ex){return 0n}var data=_memalign(16,mapped.byteLength);(growMemViews(),HEAPU8).set(new Uint8Array(mapped),data);WebGPU.Internals.bufferOnUnmaps[bufferPtr].push(()=>_free(data));return data})();return BigInt(ret)};var _emwgpuBufferMapAsync=function(bufferPtr,futureId,mode,offset,size){bufferPtr=bigintToI53Checked(bufferPtr);futureId=bigintToI53Checked(futureId);mode=bigintToI53Checked(mode);offset=bigintToI53Checked(offset);size=bigintToI53Checked(size);var buffer=WebGPU.getJsObject(bufferPtr);WebGPU.Internals.bufferOnUnmaps[bufferPtr]=[];if(size==-1)size=undefined;runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,buffer.mapAsync(mode,offset,size).then(()=>{runtimeKeepalivePop();callUserCallback(()=>{_emwgpuOnMapAsyncCompleted(futureId,1,0n)})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);var status=ex.name==="AbortError"?4:ex.name==="OperationError"?3:0;_emwgpuOnMapAsyncCompleted(futureId,status,BigInt(messagePtr));delete WebGPU.Internals.bufferOnUnmaps[bufferPtr]})}))};function _emwgpuBufferUnmap(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var buffer=WebGPU.getJsObject(bufferPtr);var onUnmap=WebGPU.Internals.bufferOnUnmaps[bufferPtr];if(!onUnmap){return}for(var i=0;i<onUnmap.length;++i){onUnmap[i]()}delete WebGPU.Internals.bufferOnUnmaps[bufferPtr];buffer.unmap()}function _emwgpuDelete(ptr){ptr=bigintToI53Checked(ptr);delete WebGPU.Internals.jsObjects[ptr]}function _emwgpuDeviceCreateBuffer(devicePtr,descriptor,bufferPtr){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);bufferPtr=bigintToI53Checked(bufferPtr);var mappedAtCreation=!!(growMemViews(),HEAPU32)[(descriptor+40)/4];var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),usage:(growMemViews(),HEAPU32)[(descriptor+24)/4],size:readI53FromI64(descriptor+32),mappedAtCreation};var device=WebGPU.getJsObject(devicePtr);var buffer;try{buffer=device.createBuffer(desc)}catch(ex){return false}WebGPU.Internals.jsObjectInsert(bufferPtr,buffer);if(mappedAtCreation){WebGPU.Internals.bufferOnUnmaps[bufferPtr]=[]}return true}function _emwgpuDeviceCreateShaderModule(devicePtr,descriptor,shaderModulePtr){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);shaderModulePtr=bigintToI53Checked(shaderModulePtr);var nextInChainPtr=Number((growMemViews(),HEAPU64)[descriptor/8]);var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),code:""};switch(sType){case 2:{desc["code"]=WebGPU.makeStringFromStringView(nextInChainPtr+16);break}}var device=WebGPU.getJsObject(devicePtr);WebGPU.Internals.jsObjectInsert(shaderModulePtr,device.createShaderModule(desc))}var _emwgpuDeviceDestroy=devicePtr=>{const device=WebGPU.getJsObject(devicePtr);device.onuncapturederror=null;device.destroy()};function _emwgpuInstanceRequestAdapter(instancePtr,futureId,options,adapterPtr){instancePtr=bigintToI53Checked(instancePtr);futureId=bigintToI53Checked(futureId);options=bigintToI53Checked(options);adapterPtr=bigintToI53Checked(adapterPtr);var opts;if(options){opts={featureLevel:WebGPU.FeatureLevel[(growMemViews(),HEAP32)[(options+8)/4]],powerPreference:WebGPU.PowerPreference[(growMemViews(),HEAP32)[(options+12)/4]],forceFallbackAdapter:!!(growMemViews(),HEAPU32)[(options+16)/4]};var nextInChainPtr=Number((growMemViews(),HEAPU64)[options/8]);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var webxrOptions=nextInChainPtr;opts.xrCompatible=!!(growMemViews(),HEAPU32)[(webxrOptions+16)/4]}}if(!("gpu"in navigator)){var sp=stackSave();var messagePtr=stringToUTF8OnStack("WebGPU not available on this browser (navigator.gpu is not available)");_emwgpuOnRequestAdapterCompleted(futureId,3,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp);return}runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,navigator.gpu.requestAdapter(opts).then(adapter=>{runtimeKeepalivePop();callUserCallback(()=>{if(adapter){WebGPU.Internals.jsObjectInsert(adapterPtr,adapter);_emwgpuOnRequestAdapterCompleted(futureId,1,BigInt(adapterPtr),0n)}else{var sp=stackSave();var messagePtr=stringToUTF8OnStack("WebGPU not available on this browser (requestAdapter returned null)");_emwgpuOnRequestAdapterCompleted(futureId,3,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp)}})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);_emwgpuOnRequestAdapterCompleted(futureId,4,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp)})}))}var _emwgpuQueueOnSubmittedWorkDone=function(queuePtr,futureId){queuePtr=bigintToI53Checked(queuePtr);futureId=bigintToI53Checked(futureId);var queue=WebGPU.getJsObject(queuePtr);runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,queue.onSubmittedWorkDone().then(()=>{runtimeKeepalivePop();callUserCallback(()=>{_emwgpuOnWorkDoneCompleted(futureId,1)})}))};var _emwgpuWaitAny=function(futurePtr,futureCount,timeoutMSPtr){futurePtr=bigintToI53Checked(futurePtr);futureCount=bigintToI53Checked(futureCount);timeoutMSPtr=bigintToI53Checked(timeoutMSPtr);return Asyncify.handleAsync(async()=>{var promises=[];if(timeoutMSPtr){var timeoutMS=(growMemViews(),HEAP32)[timeoutMSPtr/4];promises.length=futureCount+1;promises[futureCount]=new Promise(resolve=>setTimeout(resolve,timeoutMS,0))}else{promises.length=futureCount}for(var i=0;i<futureCount;++i){var futureId=readI53FromI64(futurePtr+i*8);if(!(futureId in WebGPU.Internals.futures)){return futureId}promises[i]=WebGPU.Internals.futures[futureId]}const firstResolvedFuture=await Promise.race(promises);delete WebGPU.Internals.futures[firstResolvedFuture];return firstResolvedFuture})};_emwgpuWaitAny.isAsync=true;var ENV={};var getExecutableName=()=>thisProgram||"./this.program";var getEnvStrings=()=>{if(!getEnvStrings.strings){var lang=(typeof navigator=="object"&&navigator.language||"C").replace("-","_")+".UTF-8";var env={USER:"web_user",LOGNAME:"web_user",PATH:"/",PWD:"/",HOME:"/home/web_user",LANG:lang,_:getExecutableName()};for(var x in ENV){if(ENV[x]===undefined)delete env[x];else env[x]=ENV[x]}var strings=[];for(var x in env){strings.push(`${x}=${env[x]}`)}getEnvStrings.strings=strings}return getEnvStrings.strings};function _environ_get(__environ,environ_buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(14,0,1,__environ,environ_buf);__environ=bigintToI53Checked(__environ);environ_buf=bigintToI53Checked(environ_buf);var bufSize=0;var envp=0;for(var string of getEnvStrings()){var ptr=environ_buf+bufSize;(growMemViews(),HEAPU64)[(__environ+envp)/8]=BigInt(ptr);bufSize+=stringToUTF8(string,ptr,Infinity)+1;envp+=8}return 0}function _environ_sizes_get(penviron_count,penviron_buf_size){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(15,0,1,penviron_count,penviron_buf_size);penviron_count=bigintToI53Checked(penviron_count);penviron_buf_size=bigintToI53Checked(penviron_buf_size);var strings=getEnvStrings();(growMemViews(),HEAPU64)[penviron_count/8]=BigInt(strings.length);var bufSize=0;for(var string of strings){bufSize+=lengthBytesUTF8(string)+1}(growMemViews(),HEAPU64)[penviron_buf_size/8]=BigInt(bufSize);return 0}function _fd_close(fd){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(16,0,1,fd);try{var stream=SYSCALLS.getStreamFromFD(fd);FS.close(stream);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doReadv=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=Number((growMemViews(),HEAPU64)[iov/8]);var len=Number((growMemViews(),HEAPU64)[(iov+8)/8]);iov+=16;var curr=FS.read(stream,(growMemViews(),HEAP8),ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len)break;if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_read(fd,iov,iovcnt,pnum){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(17,0,1,fd,iov,iovcnt,pnum);iov=bigintToI53Checked(iov);iovcnt=bigintToI53Checked(iovcnt);pnum=bigintToI53Checked(pnum);try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doReadv(stream,iov,iovcnt);(growMemViews(),HEAPU64)[pnum/8]=BigInt(num);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _fd_seek(fd,offset,whence,newOffset){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(18,0,1,fd,offset,whence,newOffset);offset=bigintToI53Checked(offset);newOffset=bigintToI53Checked(newOffset);try{if(isNaN(offset))return 61;var stream=SYSCALLS.getStreamFromFD(fd);FS.llseek(stream,offset,whence);(growMemViews(),HEAP64)[newOffset/8]=BigInt(stream.position);if(stream.getdents&&offset===0&&whence===0)stream.getdents=null;return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doWritev=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=Number((growMemViews(),HEAPU64)[iov/8]);var len=Number((growMemViews(),HEAPU64)[(iov+8)/8]);iov+=16;var curr=FS.write(stream,(growMemViews(),HEAP8),ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len){break}if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_write(fd,iov,iovcnt,pnum){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(19,0,1,fd,iov,iovcnt,pnum);iov=bigintToI53Checked(iov);iovcnt=bigintToI53Checked(iovcnt);pnum=bigintToI53Checked(pnum);try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doWritev(stream,iov,iovcnt);(growMemViews(),HEAPU64)[pnum/8]=BigInt(num);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _random_get(buffer,size){buffer=bigintToI53Checked(buffer);size=bigintToI53Checked(size);try{randomFill((growMemViews(),HEAPU8).subarray(buffer,buffer+size));return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _wgpuAdapterGetInfo(adapterPtr,info){adapterPtr=bigintToI53Checked(adapterPtr);info=bigintToI53Checked(info);var adapter=WebGPU.getJsObject(adapterPtr);WebGPU.fillAdapterInfoStruct(adapter.info,info);return 1}function _wgpuAdapterGetLimits(adapterPtr,limitsOutPtr){adapterPtr=bigintToI53Checked(adapterPtr);limitsOutPtr=bigintToI53Checked(limitsOutPtr);var adapter=WebGPU.getJsObject(adapterPtr);WebGPU.fillLimitStruct(adapter.limits,limitsOutPtr);return 1}function _wgpuAdapterHasFeature(adapterPtr,featureEnumValue){adapterPtr=bigintToI53Checked(adapterPtr);var adapter=WebGPU.getJsObject(adapterPtr);return adapter.features.has(WebGPU.FeatureName[featureEnumValue])}var _wgpuBufferGetSize=function(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var ret=(()=>{var buffer=WebGPU.getJsObject(bufferPtr);return buffer.size})();return BigInt(ret)};var _wgpuCommandEncoderBeginComputePass=function(encoderPtr,descriptor){encoderPtr=bigintToI53Checked(encoderPtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc;if(descriptor){desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),timestampWrites:WebGPU.makePassTimestampWrites(Number((growMemViews(),HEAPU64)[(descriptor+24)/8]))}}var commandEncoder=WebGPU.getJsObject(encoderPtr);var ptr=_emwgpuCreateComputePassEncoder(0n);WebGPU.Internals.jsObjectInsert(ptr,commandEncoder.beginComputePass(desc));return ptr})();return BigInt(ret)};function _wgpuCommandEncoderCopyBufferToBuffer(encoderPtr,srcPtr,srcOffset,dstPtr,dstOffset,size){encoderPtr=bigintToI53Checked(encoderPtr);srcPtr=bigintToI53Checked(srcPtr);srcOffset=bigintToI53Checked(srcOffset);dstPtr=bigintToI53Checked(dstPtr);dstOffset=bigintToI53Checked(dstOffset);size=bigintToI53Checked(size);var commandEncoder=WebGPU.getJsObject(encoderPtr);var src=WebGPU.getJsObject(srcPtr);var dst=WebGPU.getJsObject(dstPtr);commandEncoder.copyBufferToBuffer(src,srcOffset,dst,dstOffset,size)}var _wgpuCommandEncoderFinish=function(encoderPtr,descriptor){encoderPtr=bigintToI53Checked(encoderPtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var commandEncoder=WebGPU.getJsObject(encoderPtr);var ptr=_emwgpuCreateCommandBuffer(0n);WebGPU.Internals.jsObjectInsert(ptr,commandEncoder.finish());return ptr})();return BigInt(ret)};function _wgpuComputePassEncoderDispatchWorkgroups(passPtr,x,y,z){passPtr=bigintToI53Checked(passPtr);var pass=WebGPU.getJsObject(passPtr);pass.dispatchWorkgroups(x,y,z)}function _wgpuComputePassEncoderEnd(passPtr){passPtr=bigintToI53Checked(passPtr);var pass=WebGPU.getJsObject(passPtr);pass.end()}function _wgpuComputePassEncoderSetBindGroup(passPtr,groupIndex,groupPtr,dynamicOffsetCount,dynamicOffsetsPtr){passPtr=bigintToI53Checked(passPtr);groupPtr=bigintToI53Checked(groupPtr);dynamicOffsetCount=bigintToI53Checked(dynamicOffsetCount);dynamicOffsetsPtr=bigintToI53Checked(dynamicOffsetsPtr);var pass=WebGPU.getJsObject(passPtr);var group=WebGPU.getJsObject(groupPtr);if(dynamicOffsetCount==0){pass.setBindGroup(groupIndex,group)}else{pass.setBindGroup(groupIndex,group,(growMemViews(),HEAPU32),dynamicOffsetsPtr/4,dynamicOffsetCount)}}function _wgpuComputePassEncoderSetPipeline(passPtr,pipelinePtr){passPtr=bigintToI53Checked(passPtr);pipelinePtr=bigintToI53Checked(pipelinePtr);var pass=WebGPU.getJsObject(passPtr);var pipeline=WebGPU.getJsObject(pipelinePtr);pass.setPipeline(pipeline)}var _wgpuComputePipelineGetBindGroupLayout=function(pipelinePtr,groupIndex){pipelinePtr=bigintToI53Checked(pipelinePtr);var ret=(()=>{var pipeline=WebGPU.getJsObject(pipelinePtr);var ptr=_emwgpuCreateBindGroupLayout(0n);WebGPU.Internals.jsObjectInsert(ptr,pipeline.getBindGroupLayout(groupIndex));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateBindGroup=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{function makeEntry(entryPtr){var bufferPtr=Number((growMemViews(),HEAPU64)[(entryPtr+16)/8]);var samplerPtr=Number((growMemViews(),HEAPU64)[(entryPtr+40)/8]);var textureViewPtr=Number((growMemViews(),HEAPU64)[(entryPtr+48)/8]);var externalTexturePtr=0;WebGPU.iterateExtensions(entryPtr,{327681:ptr=>{externalTexturePtr=Number((growMemViews(),HEAPU64)[(ptr+16)/8])}});var resource;if(bufferPtr){var size=readI53FromI64(entryPtr+32);if(size==-1)size=undefined;resource={buffer:WebGPU.getJsObject(bufferPtr),offset:readI53FromI64(entryPtr+24),size}}else{resource=WebGPU.getJsObject(samplerPtr||textureViewPtr||externalTexturePtr)}return{binding:(growMemViews(),HEAPU32)[(entryPtr+8)/4],resource}}function makeEntries(count,entriesPtrs){var entries=[];for(var i=0;i<count;++i){entries.push(makeEntry(entriesPtrs+56*i))}return entries}var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),entries:makeEntries(Number((growMemViews(),HEAPU64)[(descriptor+32)/8]),Number((growMemViews(),HEAPU64)[(descriptor+40)/8]))};var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateBindGroup(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createBindGroup(desc));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateCommandEncoder=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc;if(descriptor){desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8)}}var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateCommandEncoder(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createCommandEncoder(desc));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateComputePipeline=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc=WebGPU.makeComputePipelineDesc(descriptor);var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateComputePipeline(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createComputePipeline(desc));return ptr})();return BigInt(ret)};function _wgpuInstanceHasWGSLLanguageFeature(instance,featureEnumValue){instance=bigintToI53Checked(instance);if(!("wgslLanguageFeatures"in navigator.gpu)){return false}return navigator.gpu.wgslLanguageFeatures.has(WebGPU.WGSLLanguageFeatureName[featureEnumValue])}var _wgpuQueueSubmit=function(queuePtr,commandCount,commands){queuePtr=bigintToI53Checked(queuePtr);commandCount=bigintToI53Checked(commandCount);commands=bigintToI53Checked(commands);var queue=WebGPU.getJsObject(queuePtr);var cmds=Array.from((growMemViews(),HEAP64).subarray(commands/8,(commands+commandCount*8)/8),id=>WebGPU.getJsObject(id));queue.submit(cmds)};function _wgpuQueueWriteBuffer(queuePtr,bufferPtr,bufferOffset,data,size){queuePtr=bigintToI53Checked(queuePtr);bufferPtr=bigintToI53Checked(bufferPtr);bufferOffset=bigintToI53Checked(bufferOffset);data=bigintToI53Checked(data);size=bigintToI53Checked(size);var queue=WebGPU.getJsObject(queuePtr);var buffer=WebGPU.getJsObject(bufferPtr);var subarray=(growMemViews(),HEAPU8).subarray(data,data+size);queue.writeBuffer(buffer,bufferOffset,subarray,0,size)}var Asyncify={instrumentWasmImports(imports){var importPattern=/^(invoke_.*|__asyncjs__.*)$/;for(let[x,original]of Object.entries(imports)){if(typeof original=="function"){let isAsyncifyImport=original.isAsync||importPattern.test(x);if(isAsyncifyImport){imports[x]=original=new WebAssembly.Suspending(original)}}}},instrumentFunction(original){var wrapper=(...args)=>original(...args);return wrapper},instrumentWasmExports(exports){var exportPattern=/^(wllama_start|wllama_action|main|__main_argc_argv)$/;Asyncify.asyncExports=new Set;var ret={};for(let[x,original]of Object.entries(exports)){if(typeof original=="function"){let isAsyncifyExport=exportPattern.test(x);if(isAsyncifyExport){Asyncify.asyncExports.add(original);original=Asyncify.makeAsyncFunction(original)}var wrapper=Asyncify.instrumentFunction(original);ret[x]=wrapper}else{ret[x]=original}}return ret},asyncExports:null,isAsyncExport(func){return Asyncify.asyncExports?.has(func)},handleAsync:async startAsync=>{runtimeKeepalivePush();try{return await startAsync()}finally{runtimeKeepalivePop()}},handleSleep:startAsync=>Asyncify.handleAsync(()=>new Promise(startAsync)),makeAsyncFunction(original){return WebAssembly.promising(original)}};var getCFunc=ident=>{var func=Module["_"+ident];return func};var writeArrayToMemory=(array,buffer)=>{(growMemViews(),HEAP8).set(array,buffer)};var ccall=(ident,returnType,argTypes,args,opts)=>{var toC={pointer:p=>BigInt(p),string:str=>{var ret=0;if(str!==null&&str!==undefined&&str!==0){ret=stringToUTF8OnStack(str)}return BigInt(ret)},array:arr=>{var ret=stackAlloc(arr.length);writeArrayToMemory(arr,ret);return BigInt(ret)}};function convertReturnValue(ret){if(returnType==="string"){return UTF8ToString(Number(ret))}if(returnType==="pointer")return Number(ret);if(returnType==="boolean")return Boolean(ret);return ret}var func=getCFunc(ident);var cArgs=[];var stack=0;if(args){for(var i=0;i<args.length;i++){var converter=toC[argTypes[i]];if(converter){if(stack===0)stack=stackSave();cArgs[i]=converter(args[i])}else{cArgs[i]=args[i]}}}var ret=func(...cArgs);function onDone(ret){if(stack!==0)stackRestore(stack);return convertReturnValue(ret)}var asyncMode=opts?.async;if(asyncMode)return ret.then(onDone);ret=onDone(ret);return ret};var cwrap=(ident,returnType,argTypes,opts)=>{var numericArgs=!argTypes||argTypes.every(type=>type==="number"||type==="boolean");var numericRet=returnType!=="string";if(numericRet&&numericArgs&&!opts){return getCFunc(ident)}return(...args)=>ccall(ident,returnType,argTypes,args,opts)};var FS_createPath=(...args)=>FS.createPath(...args);var FS_unlink=(...args)=>FS.unlink(...args);var FS_createLazyFile=(...args)=>FS.createLazyFile(...args);var FS_createDevice=(...args)=>FS.createDevice(...args);PThread.init();FS.createPreloadedFile=FS_createPreloadedFile;FS.preloadFile=FS_preloadFile;FS.staticInit();{initMemory();if(Module["noExitRuntime"])noExitRuntime=Module["noExitRuntime"];if(Module["preloadPlugins"])preloadPlugins=Module["preloadPlugins"];if(Module["print"])out=Module["print"];if(Module["printErr"])err=Module["printErr"];if(Module["wasmBinary"])wasmBinary=Module["wasmBinary"];if(Module["arguments"])arguments_=Module["arguments"];if(Module["thisProgram"])thisProgram=Module["thisProgram"];if(Module["preInit"]){if(typeof Module["preInit"]=="function")Module["preInit"]=[Module["preInit"]];while(Module["preInit"].length>0){Module["preInit"].shift()()}}}Module["ENV"]=ENV;Module["mmapAlloc"]=mmapAlloc;Module["wasmMemory"]=wasmMemory;Module["addRunDependency"]=addRunDependency;Module["removeRunDependency"]=removeRunDependency;Module["ccall"]=ccall;Module["cwrap"]=cwrap;Module["FS_preloadFile"]=FS_preloadFile;Module["FS_unlink"]=FS_unlink;Module["FS_createPath"]=FS_createPath;Module["FS_createDevice"]=FS_createDevice;Module["FS"]=FS;Module["FS_createDataFile"]=FS_createDataFile;Module["FS_createLazyFile"]=FS_createLazyFile;Module["MEMFS"]=MEMFS;var proxiedFunctionTable=[_proc_exit,exitOnMainThread,pthreadCreateProxied,___syscall_fcntl64,___syscall_fstat64,___syscall_getcwd,___syscall_getdents64,___syscall_ioctl,___syscall_lstat64,___syscall_newfstatat,___syscall_openat,___syscall_stat64,__mmap_js,__munmap_js,_environ_get,_environ_sizes_get,_fd_close,_fd_read,_fd_seek,_fd_write];function __asyncjs__js_file_read(path_ptr,offset,req_size,out_ptr){return Asyncify.handleAsync(async()=>await _wllama_js_file_read(UTF8ToString(Number(path_ptr)),Number(offset),Number(req_size),Number(out_ptr)))}__asyncjs__js_file_read.sig="jjjjj";var _wllama_malloc,_wllama_start,_wllama_action,_wllama_exit,_wllama_debug,_main,_malloc,_free,_emwgpuCreateBindGroup,_emwgpuCreateBindGroupLayout,_emwgpuCreateCommandBuffer,_emwgpuCreateCommandEncoder,_emwgpuCreateComputePassEncoder,_emwgpuCreateComputePipeline,_emwgpuCreateExternalTexture,_emwgpuCreatePipelineLayout,_emwgpuCreateQuerySet,_emwgpuCreateRenderBundle,_emwgpuCreateRenderBundleEncoder,_emwgpuCreateRenderPassEncoder,_emwgpuCreateRenderPipeline,_emwgpuCreateSampler,_emwgpuCreateSurface,_emwgpuCreateTexture,_emwgpuCreateTextureView,_emwgpuCreateAdapter,_emwgpuCreateBuffer,_emwgpuCreateDevice,_emwgpuCreateQueue,_emwgpuCreateShaderModule,_emwgpuOnDeviceLostCompleted,_emwgpuOnMapAsyncCompleted,_emwgpuOnRequestAdapterCompleted,_emwgpuOnRequestDeviceCompleted,_emwgpuOnWorkDoneCompleted,_emwgpuOnUncapturedError,__emscripten_tls_init,_pthread_self,_emscripten_builtin_memalign,__emscripten_thread_init,__emscripten_thread_crashed,__emscripten_run_js_on_main_thread,__emscripten_thread_free_data,__emscripten_thread_exit,__emscripten_check_mailbox,_memalign,___trap,_emscripten_stack_set_limits,__emscripten_stack_restore,__emscripten_stack_alloc,_emscripten_stack_get_current,__indirect_function_table,wasmTable;function assignWasmExports(wasmExports){_wllama_malloc=Module["_wllama_malloc"]=wasmExports["wllama_malloc"];_wllama_start=Module["_wllama_start"]=wasmExports["wllama_start"];_wllama_action=Module["_wllama_action"]=wasmExports["wllama_action"];_wllama_exit=Module["_wllama_exit"]=wasmExports["wllama_exit"];_wllama_debug=Module["_wllama_debug"]=wasmExports["wllama_debug"];_main=Module["_main"]=wasmExports["main"];_malloc=wasmExports["malloc"];_free=wasmExports["free"];_emwgpuCreateBindGroup=wasmExports["emwgpuCreateBindGroup"];_emwgpuCreateBindGroupLayout=wasmExports["emwgpuCreateBindGroupLayout"];_emwgpuCreateCommandBuffer=wasmExports["emwgpuCreateCommandBuffer"];_emwgpuCreateCommandEncoder=wasmExports["emwgpuCreateCommandEncoder"];_emwgpuCreateComputePassEncoder=wasmExports["emwgpuCreateComputePassEncoder"];_emwgpuCreateComputePipeline=wasmExports["emwgpuCreateComputePipeline"];_emwgpuCreateExternalTexture=wasmExports["emwgpuCreateExternalTexture"];_emwgpuCreatePipelineLayout=wasmExports["emwgpuCreatePipelineLayout"];_emwgpuCreateQuerySet=wasmExports["emwgpuCreateQuerySet"];_emwgpuCreateRenderBundle=wasmExports["emwgpuCreateRenderBundle"];_emwgpuCreateRenderBundleEncoder=wasmExports["emwgpuCreateRenderBundleEncoder"];_emwgpuCreateRenderPassEncoder=wasmExports["emwgpuCreateRenderPassEncoder"];_emwgpuCreateRenderPipeline=wasmExports["emwgpuCreateRenderPipeline"];_emwgpuCreateSampler=wasmExports["emwgpuCreateSampler"];_emwgpuCreateSurface=wasmExports["emwgpuCreateSurface"];_emwgpuCreateTexture=wasmExports["emwgpuCreateTexture"];_emwgpuCreateTextureView=wasmExports["emwgpuCreateTextureView"];_emwgpuCreateAdapter=wasmExports["emwgpuCreateAdapter"];_emwgpuCreateBuffer=wasmExports["emwgpuCreateBuffer"];_emwgpuCreateDevice=wasmExports["emwgpuCreateDevice"];_emwgpuCreateQueue=wasmExports["emwgpuCreateQueue"];_emwgpuCreateShaderModule=wasmExports["emwgpuCreateShaderModule"];_emwgpuOnDeviceLostCompleted=wasmExports["emwgpuOnDeviceLostCompleted"];_emwgpuOnMapAsyncCompleted=wasmExports["emwgpuOnMapAsyncCompleted"];_emwgpuOnRequestAdapterCompleted=wasmExports["emwgpuOnRequestAdapterCompleted"];_emwgpuOnRequestDeviceCompleted=wasmExports["emwgpuOnRequestDeviceCompleted"];_emwgpuOnWorkDoneCompleted=wasmExports["emwgpuOnWorkDoneCompleted"];_emwgpuOnUncapturedError=wasmExports["emwgpuOnUncapturedError"];__emscripten_tls_init=wasmExports["_emscripten_tls_init"];_pthread_self=wasmExports["pthread_self"];_emscripten_builtin_memalign=wasmExports["emscripten_builtin_memalign"];__emscripten_thread_init=wasmExports["_emscripten_thread_init"];__emscripten_thread_crashed=wasmExports["_emscripten_thread_crashed"];__emscripten_run_js_on_main_thread=wasmExports["_emscripten_run_js_on_main_thread"];__emscripten_thread_free_data=wasmExports["_emscripten_thread_free_data"];__emscripten_thread_exit=wasmExports["_emscripten_thread_exit"];__emscripten_check_mailbox=wasmExports["_emscripten_check_mailbox"];_memalign=wasmExports["memalign"];___trap=wasmExports["__trap"];_emscripten_stack_set_limits=wasmExports["emscripten_stack_set_limits"];__emscripten_stack_restore=wasmExports["_emscripten_stack_restore"];__emscripten_stack_alloc=wasmExports["_emscripten_stack_alloc"];_emscripten_stack_get_current=wasmExports["emscripten_stack_get_current"];__indirect_function_table=wasmTable=wasmExports["__indirect_function_table"]}var wasmImports;function assignWasmImports(){wasmImports={__asyncjs__js_file_read,__pthread_create_js:___pthread_create_js,__syscall_fcntl64:___syscall_fcntl64,__syscall_getcwd:___syscall_getcwd,__syscall_getdents64:___syscall_getdents64,__syscall_ioctl:___syscall_ioctl,__syscall_openat:___syscall_openat,__syscall_stat64:___syscall_stat64,_abort_js:__abort_js,_emscripten_init_main_thread_js:__emscripten_init_main_thread_js,_emscripten_notify_mailbox_postmessage:__emscripten_notify_mailbox_postmessage,_emscripten_receive_on_main_thread_js:__emscripten_receive_on_main_thread_js,_emscripten_thread_cleanup:__emscripten_thread_cleanup,_emscripten_thread_mailbox_await:__emscripten_thread_mailbox_await,_emscripten_thread_set_strongref:__emscripten_thread_set_strongref,_localtime_js:__localtime_js,_mmap_js:__mmap_js,_munmap_js:__munmap_js,_tzset_js:__tzset_js,clock_time_get:_clock_time_get,emscripten_check_blocking_allowed:_emscripten_check_blocking_allowed,emscripten_date_now:_emscripten_date_now,emscripten_exit_with_live_runtime:_emscripten_exit_with_live_runtime,emscripten_get_callstack:_emscripten_get_callstack,emscripten_get_heap_max:_emscripten_get_heap_max,emscripten_get_now:_emscripten_get_now,emscripten_has_asyncify:_emscripten_has_asyncify,emscripten_num_logical_cores:_emscripten_num_logical_cores,emscripten_resize_heap:_emscripten_resize_heap,emwgpuAdapterRequestDevice:_emwgpuAdapterRequestDevice,emwgpuBufferDestroy:_emwgpuBufferDestroy,emwgpuBufferGetConstMappedRange:_emwgpuBufferGetConstMappedRange,emwgpuBufferMapAsync:_emwgpuBufferMapAsync,emwgpuBufferUnmap:_emwgpuBufferUnmap,emwgpuDelete:_emwgpuDelete,emwgpuDeviceCreateBuffer:_emwgpuDeviceCreateBuffer,emwgpuDeviceCreateShaderModule:_emwgpuDeviceCreateShaderModule,emwgpuDeviceDestroy:_emwgpuDeviceDestroy,emwgpuInstanceRequestAdapter:_emwgpuInstanceRequestAdapter,emwgpuQueueOnSubmittedWorkDone:_emwgpuQueueOnSubmittedWorkDone,emwgpuWaitAny:_emwgpuWaitAny,environ_get:_environ_get,environ_sizes_get:_environ_sizes_get,exit:_exit,fd_close:_fd_close,fd_read:_fd_read,fd_seek:_fd_seek,fd_write:_fd_write,memory:wasmMemory,random_get:_random_get,wgpuAdapterGetInfo:_wgpuAdapterGetInfo,wgpuAdapterGetLimits:_wgpuAdapterGetLimits,wgpuAdapterHasFeature:_wgpuAdapterHasFeature,wgpuBufferGetSize:_wgpuBufferGetSize,wgpuCommandEncoderBeginComputePass:_wgpuCommandEncoderBeginComputePass,wgpuCommandEncoderCopyBufferToBuffer:_wgpuCommandEncoderCopyBufferToBuffer,wgpuCommandEncoderFinish:_wgpuCommandEncoderFinish,wgpuComputePassEncoderDispatchWorkgroups:_wgpuComputePassEncoderDispatchWorkgroups,wgpuComputePassEncoderEnd:_wgpuComputePassEncoderEnd,wgpuComputePassEncoderSetBindGroup:_wgpuComputePassEncoderSetBindGroup,wgpuComputePassEncoderSetPipeline:_wgpuComputePassEncoderSetPipeline,wgpuComputePipelineGetBindGroupLayout:_wgpuComputePipelineGetBindGroupLayout,wgpuDeviceCreateBindGroup:_wgpuDeviceCreateBindGroup,wgpuDeviceCreateCommandEncoder:_wgpuDeviceCreateCommandEncoder,wgpuDeviceCreateComputePipeline:_wgpuDeviceCreateComputePipeline,wgpuInstanceHasWGSLLanguageFeature:_wgpuInstanceHasWGSLLanguageFeature,wgpuQueueSubmit:_wgpuQueueSubmit,wgpuQueueWriteBuffer:_wgpuQueueWriteBuffer}}function applySignatureConversions(wasmExports){wasmExports=Object.assign({},wasmExports);var makeWrapper___PP=f=>(a0,a1,a2)=>f(a0,BigInt(a1?a1:0),BigInt(a2?a2:0));var makeWrapper_pp=f=>a0=>Number(f(BigInt(a0)));var makeWrapper__p=f=>a0=>f(BigInt(a0));var makeWrapper_p=f=>()=>Number(f());var makeWrapper_ppp=f=>(a0,a1)=>Number(f(BigInt(a0),BigInt(a1)));var makeWrapper__p_____=f=>(a0,a1,a2,a3,a4,a5)=>f(BigInt(a0),a1,a2,a3,a4,a5);var makeWrapper___p_p_=f=>(a0,a1,a2,a3,a4)=>f(a0,BigInt(a1),a2,BigInt(a3),a4);var makeWrapper__pp=f=>(a0,a1)=>f(BigInt(a0),BigInt(a1));wasmExports["main"]=makeWrapper___PP(wasmExports["main"]);wasmExports["malloc"]=makeWrapper_pp(wasmExports["malloc"]);wasmExports["free"]=makeWrapper__p(wasmExports["free"]);wasmExports["pthread_self"]=makeWrapper_p(wasmExports["pthread_self"]);wasmExports["emscripten_builtin_memalign"]=makeWrapper_ppp(wasmExports["emscripten_builtin_memalign"]);wasmExports["_emscripten_thread_init"]=makeWrapper__p_____(wasmExports["_emscripten_thread_init"]);wasmExports["_emscripten_run_js_on_main_thread"]=makeWrapper___p_p_(wasmExports["_emscripten_run_js_on_main_thread"]);wasmExports["_emscripten_thread_free_data"]=makeWrapper__p(wasmExports["_emscripten_thread_free_data"]);wasmExports["_emscripten_thread_exit"]=makeWrapper__p(wasmExports["_emscripten_thread_exit"]);wasmExports["memalign"]=makeWrapper_ppp(wasmExports["memalign"]);wasmExports["emscripten_stack_set_limits"]=makeWrapper__pp(wasmExports["emscripten_stack_set_limits"]);wasmExports["_emscripten_stack_restore"]=makeWrapper__p(wasmExports["_emscripten_stack_restore"]);wasmExports["_emscripten_stack_alloc"]=makeWrapper_pp(wasmExports["_emscripten_stack_alloc"]);wasmExports["emscripten_stack_get_current"]=makeWrapper_p(wasmExports["emscripten_stack_get_current"]);return wasmExports}async function callMain(){var entryFunction=_main;var argc=0;var argv=0;try{var ret=entryFunction(argc,BigInt(argv));ret=await ret;exitJS(ret,true);return ret}catch(e){return handleException(e)}}function run(){if(runDependencies>0){dependenciesFulfilled=run;return}if(ENVIRONMENT_IS_PTHREAD){initRuntime();return}preRun();if(runDependencies>0){dependenciesFulfilled=run;return}async function doRun(){Module["calledRun"]=true;if(ABORT)return;initRuntime();preMain();Module["onRuntimeInitialized"]?.();var noInitialRun=Module["noInitialRun"]||false;if(!noInitialRun)await callMain();postRun()}if(Module["setStatus"]){Module["setStatus"]("Running...");setTimeout(()=>{setTimeout(()=>Module["setStatus"](""),1);doRun()},1)}else{doRun()}}var wasmExports;if(!ENVIRONMENT_IS_PTHREAD){createWasm();run()}\n';

// src/worker.ts
var FILE_READ_REQ_EVENT = "fs.read_req";
var JSPI_STUB = `
if (!WebAssembly.Suspending) {
  // JSPI not available - stubs that keep the import/export tables valid.
  // Suspending wraps imports: identity is fine since async imports won't be called.
  WebAssembly.Suspending = function (fn) {
    // console.log(fn.toString());
    return fn;
  };
  // promising wraps exports: must return a Promise so ccall's ret.then() works.
  WebAssembly.promising = function (fn) {
    return function (...args) {
      try {
        return Promise.resolve(fn(...args));
      } catch (e) {
        return Promise.reject(e);
      }
    };
  };
}
`;
var ProxyToWorker = class {
  // filename -> Blob for async reads
  constructor(resources, nbThread, suppressNativeLog, logger) {
    __publicField(this, "resources");
    __publicField(this, "logger");
    __publicField(this, "suppressNativeLog");
    __publicField(this, "taskQueue", []);
    __publicField(this, "taskId", 1);
    __publicField(this, "resultQueue", []);
    __publicField(this, "busy", false);
    // is the work loop is running?
    __publicField(this, "worker");
    __publicField(this, "multiThread");
    __publicField(this, "nbThread");
    __publicField(this, "useAsyncFile");
    __publicField(this, "fileBlobs", /* @__PURE__ */ new Map());
    this.resources = resources;
    this.nbThread = nbThread;
    this.multiThread = nbThread > 0;
    this.logger = logger;
    this.suppressNativeLog = suppressNativeLog;
    this.useAsyncFile = canUseAsyncFileRead(resources.compat);
  }
  getModuleCode() {
    return __async(this, null, function* () {
      if (!this.resources.jsPath) {
        if (this.resources.compat) {
          throw new Error(
            "compat mode is enabled but no jsPath was provided. Pass a worker JS via setCompat() or install @wllama/wllama-compat."
          );
        }
        return WLLAMA_EMSCRIPTEN_CODE;
      } else if (this.resources.jsPath.code) {
        return this.resources.jsPath.code;
      } else if (isString(this.resources.jsPath)) {
        const response = yield fetch(this.resources.jsPath);
        if (!response.ok) {
          throw new Error(
            `Failed to fetch worker code from ${this.resources.jsPath}`
          );
        }
        return yield response.text();
      } else {
        throw new Error("No JS code provided for worker");
      }
    });
  }
  moduleInit(ggufFiles) {
    return __async(this, null, function* () {
      let moduleCode = JSPI_STUB + (yield this.getModuleCode());
      if (this.resources.noWebGPU) {
        moduleCode = 'try{Object.defineProperty(WorkerNavigator.prototype,"gpu",{get:()=>({requestAdapter:async()=>null})});}catch(e){}' + moduleCode;
      }
      let mainModuleCode = moduleCode.replace("var Module", "var ___Module");
      const runOptions = {
        pathConfig: {
          "wllama.wasm": this.resources.wasmPath
        },
        nbThread: this.nbThread,
        compat: this.resources.compat
      };
      const completeCode = [
        `const RUN_OPTIONS = ${JSON.stringify(runOptions)};`,
        `function wModuleInit() { ${mainModuleCode}; return Module; }`,
        LLAMA_CPP_WORKER_CODE
      ].join(";\n\n");
      this.worker = createWorker(completeCode);
      this.worker.onmessage = this.onRecvMsg.bind(this);
      this.worker.onerror = this.logger.error;
      const res = yield this.pushTask({
        verb: "module.init",
        args: [
          new Blob([moduleCode], { type: "text/javascript" }),
          this.useAsyncFile
        ],
        callbackId: this.taskId++
      });
      const nativeFiles = [];
      for (const file of ggufFiles) {
        const needAllocBuffer = !this.useAsyncFile;
        const id = yield this.fileAlloc(
          file.name,
          file.blob.size,
          needAllocBuffer
        );
        nativeFiles.push(__spreadValues({ id }, file));
        if (this.useAsyncFile) {
          this.fileBlobs.set(file.name, file.blob);
        }
      }
      if (!this.useAsyncFile) {
        yield Promise.all(
          nativeFiles.map((file) => {
            return this.fileWrite(file.id, file.blob);
          })
        );
      }
      return res;
    });
  }
  wllamaStart() {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "wllama.start",
        args: [],
        callbackId: this.taskId++
      });
      const parsedResult = this.parseResult(result);
      return parsedResult;
    });
  }
  wllamaAction(name, body) {
    return __async(this, null, function* () {
      const encodedMsg = glueSerialize(body);
      const result = yield this.pushTask({
        verb: "wllama.action",
        args: [name, encodedMsg],
        callbackId: this.taskId++
      });
      const parsedResult = glueDeserialize(result);
      if (parsedResult._name === "erro_evt") {
        throw new WllamaRuntimeError(parsedResult.message, "");
      }
      return parsedResult;
    });
  }
  /** Take a slot save file out of the worker's in-memory FS (the file is removed) */
  slotFileRead(fileName) {
    return __async(this, null, function* () {
      return yield this.pushTask({
        verb: "fs.slot_read",
        args: [fileName],
        callbackId: this.taskId++
      });
    });
  }
  /** Put a slot save file into the worker's in-memory FS */
  slotFileWrite(fileName, data) {
    return __async(this, null, function* () {
      yield this.pushTask({
        verb: "fs.slot_write",
        args: [fileName, data],
        callbackId: this.taskId++
      });
    });
  }
  slotFileRemove(fileName) {
    return __async(this, null, function* () {
      yield this.pushTask({
        verb: "fs.slot_remove",
        args: [fileName],
        callbackId: this.taskId++
      });
    });
  }
  wllamaExit() {
    return __async(this, null, function* () {
      if (this.worker) {
        this.worker.terminate();
      }
    });
  }
  wllamaDebug() {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "wllama.debug",
        args: [],
        callbackId: this.taskId++
      });
      return JSON.parse(result);
    });
  }
  ///////////////////////////////////////
  /**
   * Allocate a new file in heapfs
   * @returns fileId, to be used by fileWrite()
   */
  fileAlloc(fileName, size, allocBuffer) {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "fs.alloc",
        args: [fileName, size, allocBuffer],
        callbackId: this.taskId++
      });
      return result.fileId;
    });
  }
  /**
   * Write a Blob to heapfs
   */
  fileWrite(fileId, blob) {
    return __async(this, null, function* () {
      const reader = blob.stream().getReader();
      let offset = 0;
      while (true) {
        const { done, value } = yield reader.read();
        if (done) break;
        const size = value.byteLength;
        yield this.pushTask(
          {
            verb: "fs.write",
            args: [fileId, value, offset],
            callbackId: this.taskId++
          },
          // @ts-ignore Type 'ArrayBufferLike' is not assignable to type 'ArrayBuffer'
          [value.buffer]
        );
        offset += size;
      }
    });
  }
  fileReadResponse(name, offset, size) {
    return __async(this, null, function* () {
      var _a;
      try {
        const blob = this.fileBlobs.get(name);
        if (!blob) {
          throw new Error(`blob not found for name="${name}"`);
        }
        const chunk = blob.slice(offset, offset + size);
        const buffer = yield chunk.arrayBuffer();
        this.worker.postMessage(
          { verb: "fs.read_res", args: [buffer] },
          { transfer: [buffer] }
        );
      } catch (err) {
        this.logger.error("fileReadResponse failed, terminating worker:", err);
        (_a = this.worker) == null ? void 0 : _a.terminate();
        this.worker = void 0;
        this.abort(`File read failed: ${err}`, err.stack || "");
      }
    });
  }
  /**
   * Parse JSON result returned by cpp code.
   * Throw new Error if "__exception" is present in the response
   *
   * TODO: get rid of this function once everything is migrated to Glue
   */
  parseResult(result) {
    const parsedResult = JSON.parse(result);
    if (parsedResult && parsedResult["error"]) {
      throw new WllamaRuntimeError("Unknown error, please see console.log", "");
    }
    return parsedResult;
  }
  /**
   * Push a new task to taskQueue
   */
  pushTask(param, buffers) {
    return new Promise((resolve, reject) => {
      this.taskQueue.push({ resolve, reject, param, buffers });
      this.runTaskLoop();
    });
  }
  /**
   * Main loop for processing tasks
   */
  runTaskLoop() {
    return __async(this, null, function* () {
      var _a;
      if (this.busy) {
        return;
      }
      this.busy = true;
      while (true) {
        const task = this.taskQueue.shift();
        if (!task) break;
        this.resultQueue.push(task);
        this.worker.postMessage(
          task.param,
          isSafariMobile() ? void 0 : {
            transfer: (_a = task.buffers) != null ? _a : []
          }
        );
      }
      this.busy = false;
    });
  }
  /**
   * Handle messages from worker
   */
  onRecvMsg(e) {
    if (!e.data) return;
    const { verb, args } = e.data;
    const isCompatBuild = this.resources.compat;
    if (verb && verb.startsWith("console.")) {
      if (this.suppressNativeLog) {
        return;
      }
      if (verb.endsWith("debug")) this.logger.debug(...args);
      if (verb.endsWith("log")) this.logger.log(...args);
      if (verb.endsWith("warn")) this.logger.warn(...args);
      if (verb.endsWith("error")) this.logger.error(...args);
      return;
    } else if (verb === "signal.abort") {
      const [signalType, message, rawStack, originalErr] = args;
      if (originalErr) {
        this.logger.error(originalErr);
      }
      (() => __async(this, null, function* () {
        let stack = "";
        let newMsg = message.replace(
          "Build with -sASSERTIONS for more info.",
          ""
        );
        if (signalType === "abort") {
          newMsg = `(ABORT) ${newMsg}`;
          stack = rawStack.replace(/\|/g, "\n");
        } else if (signalType === "exception") {
          stack = rawStack;
        }
        const decoded = yield Debug.decodeStackTrace(stack, isCompatBuild);
        this.logger.error(`Stack trace (${signalType}):
` + decoded);
        this.abort(newMsg, decoded);
      }))();
      return;
    }
    if (verb === FILE_READ_REQ_EVENT) {
      const [name, offset, size] = args;
      this.fileReadResponse(name, offset, size).catch(() => {
      });
      return;
    }
    const { callbackId, result, err } = e.data;
    if (callbackId) {
      const idx = this.resultQueue.findIndex(
        (t) => t.param.callbackId === callbackId
      );
      if (idx !== -1) {
        const waitingTask = this.resultQueue.splice(idx, 1)[0];
        if (err) waitingTask.reject(err);
        else waitingTask.resolve(result);
      } else {
        this.logger.error(
          `Cannot find waiting task with callbackId = ${callbackId}`
        );
      }
    }
  }
  abort(text, stack) {
    const error = new WllamaRuntimeError(
      text.length == 0 ? "(unknown error)" : text,
      stack
    );
    while (this.resultQueue.length > 0) {
      const waitingTask = this.resultQueue.pop();
      if (!waitingTask) break;
      waitingTask.reject(error);
    }
    while (this.taskQueue.length > 0) {
      const pendingTask = this.taskQueue.pop();
      if (!pendingTask) break;
      pendingTask.reject(error);
    }
  }
};

// src/huggingface.ts
var HF_BASE = "https://huggingface.co";
var DEFAULT_QUANTS = ["Q4_K_M", "Q8_0"];
function fetchRepoFiles(repo, token) {
  return __async(this, null, function* () {
    var _a;
    const url = `${HF_BASE}/api/models/${repo}/tree/main?recursive=true`;
    const headers = { Accept: "application/json" };
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    const res = yield fetch(url, { headers });
    if (!res.ok) {
      let msg = res.statusText;
      try {
        msg = (_a = (yield res.json()).error) != null ? _a : msg;
      } catch (e) {
      }
      throw new Error(`HF API error (${res.status}): ${msg}`);
    }
    return res.json();
  });
}
function firstShardPath(files, path) {
  const m = path.match(/^(.+)-(\d{5})-of-(\d{5})\.gguf$/i);
  if (!m) return path;
  const first = `${m[1]}-00001-of-${m[3]}.gguf`;
  return files.some((f) => f.path === first) ? first : path;
}
function selectFile(files, quant, mmprojOnly) {
  const candidates = files.filter((f) => {
    if (f.type !== "file" || !f.path.toLowerCase().endsWith(".gguf"))
      return false;
    const ismmproj = f.path.toLowerCase().includes("mmproj");
    return mmprojOnly ? ismmproj : !ismmproj;
  });
  if (candidates.length === 0) return null;
  if (quant) {
    const upper = quant.toUpperCase();
    const match = candidates.find((f) => f.path.toUpperCase().includes(upper));
    if (match) return firstShardPath(candidates, match.path);
    return null;
  }
  for (const q of DEFAULT_QUANTS) {
    const match = candidates.find((f) => f.path.toUpperCase().includes(q));
    if (match) return firstShardPath(candidates, match.path);
  }
  return firstShardPath(candidates, candidates[0].path);
}
function getHFModelSource(config) {
  return __async(this, null, function* () {
    const { repo, file, quant, mmprojFile, mmprojQuant, hfToken } = config;
    const files = yield fetchRepoFiles(repo, hfToken);
    const modelPath = file != null ? file : selectFile(files, quant, false);
    if (!modelPath) {
      throw new Error(`No GGUF file found in repo "${repo}"`);
    }
    const source = {
      url: `${HF_BASE}/${repo}/resolve/main/${modelPath}`
    };
    if (mmprojFile || mmprojQuant !== void 0) {
      const mmpath = mmprojFile != null ? mmprojFile : selectFile(files, mmprojQuant, true);
      if (mmpath) {
        source.mmprojUrl = `${HF_BASE}/${repo}/resolve/main/${mmpath}`;
      }
    }
    if (hfToken) {
      const params = new URLSearchParams({ token: hfToken });
      source.url += `?${params}`;
      if (source.mmprojUrl) {
        source.mmprojUrl += `?${params}`;
      }
    }
    return source;
  });
}
function getHFFileSHA256(url, headers) {
  return __async(this, null, function* () {
    if (!url.includes("/resolve/")) return void 0;
    const rawUrl = url.replace("/resolve/", "/raw/");
    try {
      const text = yield fetch(rawUrl, { headers }).then((r) => r.text());
      const match = text.match(/^oid sha256:([0-9a-f]{64})$/m);
      return match ? match[1] : void 0;
    } catch (e) {
      return void 0;
    }
  });
}

// src/storage/opfs.ts
var OPFSBackend = class {
  isSupported() {
    var _a;
    return typeof navigator !== "undefined" && "storage" in navigator && !!((_a = navigator.storage) == null ? void 0 : _a.getDirectory);
  }
  read(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        const fileHandle = yield cacheDir.getFileHandle(key);
        return yield fileHandle.getFile();
      } catch (e) {
        return null;
      }
    });
  }
  write(key, stream) {
    return __async(this, null, function* () {
      const writable = yield openWritable(key);
      yield writable.truncate(0);
      const reader = stream.getReader();
      try {
        while (true) {
          const { done, value } = yield reader.read();
          if (done) break;
          yield writable.write(value);
        }
      } finally {
        yield writable.close();
      }
    });
  }
  getSize(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        const fileHandle = yield cacheDir.getFileHandle(key);
        const file = yield fileHandle.getFile();
        return file.size;
      } catch (e) {
        return -1;
      }
    });
  }
  list() {
    return __async(this, null, function* () {
      const cacheDir = yield getCacheDir();
      const result = [];
      try {
        for (var iter = __forAwait(cacheDir.entries()), more, temp, error; more = !(temp = yield iter.next()).done; more = false) {
          const [name, handle] = temp.value;
          if (handle.kind === "file") {
            const file = yield handle.getFile();
            result.push({ key: name, size: file.size });
          }
        }
      } catch (temp) {
        error = [temp];
      } finally {
        try {
          more && (temp = iter.return) && (yield temp.call(iter));
        } finally {
          if (error)
            throw error[0];
        }
      }
      return result;
    });
  }
  delete(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        yield cacheDir.removeEntry(key);
      } catch (e) {
        if ((e == null ? void 0 : e.name) !== "NotFoundError") throw e;
      }
    });
  }
};
function getCacheDir() {
  return __async(this, null, function* () {
    const opfsRoot = yield navigator.storage.getDirectory();
    return opfsRoot.getDirectoryHandle("cache", { create: true });
  });
}
function openWritable(fileName) {
  return __async(this, null, function* () {
    const worker = createWorker(OPFS_UTILS_WORKER_CODE);
    let pResolve;
    let pReject;
    worker.onmessage = (e) => {
      if (e.data.ok) pResolve(null);
      else if (e.data.err) pReject(e.data.err);
    };
    worker.onerror = (e) => {
      var _a;
      return pReject == null ? void 0 : pReject((_a = e.message) != null ? _a : e);
    };
    const workerExec = (data) => new Promise((resolve, reject) => {
      pResolve = resolve;
      pReject = reject;
      worker.postMessage(
        data,
        isSafariMobile() ? void 0 : { transfer: "buf" in data && data.buf ? [data.buf.buffer] : [] }
      );
    });
    yield workerExec({ action: "open", filename: fileName });
    return {
      truncate: () => __async(this, null, function* () {
      }),
      write: (value) => workerExec({ action: "write", buf: value }),
      close: () => __async(this, null, function* () {
        yield workerExec({ action: "close" });
        worker.terminate();
      })
    };
  });
}

// src/storage/cos.ts
function makeHash(key) {
  return { algorithm: "SHA-256", value: key };
}
var COSInternalBackend = class {
  isSupported() {
    return typeof navigator !== "undefined" && "crossOriginStorage" in navigator;
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  read(key) {
    return __async(this, null, function* () {
      try {
        const handle = yield navigator.crossOriginStorage.requestFileHandle(
          makeHash(key)
        );
        return handle.getFile();
      } catch (e) {
        return null;
      }
    });
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  write(key, stream) {
    return __async(this, null, function* () {
      const handle = yield navigator.crossOriginStorage.requestFileHandle(
        makeHash(key),
        { create: true }
      );
      const writable = yield handle.createWritable();
      const reader = stream.getReader();
      try {
        while (true) {
          const { done, value } = yield reader.read();
          if (done) break;
          yield writable.write(value);
        }
      } finally {
        yield writable.close();
      }
    });
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  getSize(key) {
    return __async(this, null, function* () {
      try {
        const handle = yield navigator.crossOriginStorage.requestFileHandle(
          makeHash(key)
        );
        const file = yield handle.getFile();
        return file.size;
      } catch (e) {
        return -1;
      }
    });
  }
  list() {
    return __async(this, null, function* () {
      throw new Error("not implemented");
    });
  }
  delete(_key) {
    return __async(this, null, function* () {
      throw new Error("not implemented");
    });
  }
};
var COSBackend = class {
  constructor() {
    __publicField(this, "cos", new COSInternalBackend());
    __publicField(this, "priv", new OPFSBackend());
  }
  isSupported() {
    return this.priv.isSupported();
  }
  read(key, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        const blob = yield this.cos.read(hint.sha256);
        if (blob) return blob;
      }
      return this.priv.read(key);
    });
  }
  write(key, stream, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        yield this.cos.write(hint.sha256, stream);
      } else {
        yield this.priv.write(key, stream);
      }
    });
  }
  getSize(key, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        const size = yield this.cos.getSize(hint.sha256);
        if (size !== -1) return size;
      }
      return this.priv.getSize(key);
    });
  }
  list() {
    return __async(this, null, function* () {
      return this.priv.list();
    });
  }
  delete(key) {
    return __async(this, null, function* () {
      return this.priv.delete(key);
    });
  }
};

// src/cache-manager.ts
var PREFIX_METADATA = "__metadata__";
var POLYFILL_ETAG = "polyfill_for_older_version";
function hintFromMetadata(metadata) {
  if (!metadata) return void 0;
  if (metadata.sha256) return { sha256: metadata.sha256 };
  return void 0;
}
var CacheManager = class {
  /**
   * @param backends Array of storage backends to use, in order of preference ; if first is available, use it, otherwise try the next one.
   */
  constructor(backends = [new COSBackend()]) {
    __publicField(this, "sb");
    for (const backend of backends) {
      if (backend.isSupported()) {
        this.sb = backend;
        return;
      }
    }
    throw new Error("No supported storage backend found");
  }
  /**
   * Convert a given URL into a storage key.
   *
   * Format: `${hashSHA1(fullURL)}_${fileName}`
   */
  getNameFromURL(url) {
    return __async(this, null, function* () {
      return urlToFileName(url, "");
    });
  }
  /**
   * @deprecated Use `download()` instead
   *
   * Write a new file to cache. This will overwrite existing file.
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   */
  write(name, stream, metadata) {
    return __async(this, null, function* () {
      yield this.sb.write(name, stream);
      yield this.writeMetadata(name, metadata);
    });
  }
  download(_0) {
    return __async(this, arguments, function* (url, options = {}) {
      var _a, _b, _c, _d;
      const fileKey = yield urlToFileName(url, "");
      const sha256 = yield getHFFileSHA256(url, (_a = options.headers) != null ? _a : {});
      const hint = sha256 ? { sha256 } : void 0;
      const cachedSize = yield this.sb.getSize(fileKey, hint);
      if (cachedSize !== -1) {
        const metadata2 = yield this.readMetadata(fileKey);
        if ((metadata2 == null ? void 0 : metadata2.originalURL) === url && metadata2.originalSize === cachedSize) {
          return;
        }
        const head = yield fetch(url, __spreadValues({
          method: "HEAD"
        }, options.headers ? { headers: options.headers } : {}));
        const originalSize = parseInt(
          (_b = head.headers.get("content-length")) != null ? _b : "0",
          10
        );
        const etag2 = (head.headers.get("etag") || "").replace(
          /[^A-Za-z0-9]/g,
          ""
        );
        if (originalSize > 0 && originalSize === cachedSize) {
          yield this.writeMetadata(fileKey, __spreadValues({
            originalURL: url,
            originalSize,
            etag: etag2,
            sha256
          }, (_c = options.metadataAdditional) != null ? _c : {}));
          return;
        }
        yield this.sb.delete(fileKey);
        yield this.sb.delete(`${PREFIX_METADATA}${fileKey}`);
      }
      const response = yield fetch(url, __spreadValues(__spreadValues({}, options.headers ? { headers: options.headers } : {}), options.signal ? { signal: options.signal } : {}));
      if (!response.ok || !response.body) {
        throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
      }
      const contentLength = response.headers.get("content-length");
      const etag = (response.headers.get("etag") || "").replace(
        /[^A-Za-z0-9]/g,
        ""
      );
      const total = parseInt(contentLength != null ? contentLength : "0", 10);
      const progressCallback = options.progressCallback;
      let loaded = 0;
      let lastProgressAt = 0;
      const progressStream = new TransformStream({
        transform(chunk, controller) {
          loaded += chunk.byteLength;
          if (progressCallback) {
            const now = Date.now();
            if (now - lastProgressAt > 100) {
              lastProgressAt = now;
              progressCallback({ loaded, total });
            }
          }
          controller.enqueue(chunk);
        },
        flush() {
          progressCallback == null ? void 0 : progressCallback({ loaded, total: total || loaded });
        }
      });
      const metadata = __spreadValues({
        originalURL: url,
        originalSize: total,
        etag
      }, (_d = options.metadataAdditional) != null ? _d : {});
      if (sha256) {
        metadata.sha256 = sha256;
      }
      yield this.sb.write(
        fileKey,
        response.body.pipeThrough(progressStream),
        hint
      );
      yield this.writeMetadata(fileKey, metadata);
    });
  }
  /**
   * Open a file in cache for reading
   *
   * @param nameOrURL The file name returned by `getNameFromURL()` or `list()`, or the original URL of the remote file
   * @returns Blob, or null if file does not exist
   */
  open(nameOrURL) {
    return __async(this, null, function* () {
      const hint1 = hintFromMetadata(yield this.getMetadata(nameOrURL));
      const direct = yield this.sb.read(nameOrURL, hint1);
      if (direct) return direct;
      const key = yield urlToFileName(nameOrURL, "");
      const hint2 = hintFromMetadata(yield this.getMetadata(key));
      return this.sb.read(key, hint2);
    });
  }
  /**
   * Get the size of a file in stored cache
   *
   * NOTE: in case the download is stopped mid-way (i.e. user close browser tab), the file maybe corrupted, size maybe different from `metadata.originalSize`
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   * @returns number of bytes, or -1 if file does not exist
   */
  getSize(name) {
    return __async(this, null, function* () {
      const hint = hintFromMetadata(yield this.getMetadata(name));
      return this.sb.getSize(name, hint);
    });
  }
  /**
   * Get metadata of a cached file
   */
  getMetadata(name) {
    return __async(this, null, function* () {
      const metadata = yield this.readMetadata(name);
      if (metadata) return metadata;
      const cachedSize = yield this.sb.getSize(name);
      return cachedSize > 0 ? (
        // files created by older version of wllama don't have metadata; polyfill it
        {
          etag: POLYFILL_ETAG,
          originalSize: cachedSize,
          originalURL: ""
        }
      ) : (
        // cached file not found
        null
      );
    });
  }
  /**
   * Same as `getMetadata()`, but without polyfill. Returns null if the file has no metadata.
   */
  readMetadata(name) {
    return __async(this, null, function* () {
      const blob = yield this.sb.read(`${PREFIX_METADATA}${name}`);
      if (!blob) return null;
      try {
        return yield new Response(blob).json();
      } catch (e) {
        return null;
      }
    });
  }
  /**
   * List all files currently in cache
   */
  list() {
    return __async(this, null, function* () {
      const all = yield this.sb.list();
      const metadataMap = {};
      for (const { key } of all) {
        if (key.startsWith(PREFIX_METADATA)) {
          const blob = yield this.sb.read(key);
          if (blob) {
            const meta = yield new Response(blob).json().catch(() => null);
            metadataMap[key.slice(PREFIX_METADATA.length)] = meta;
          }
        }
      }
      const result = [];
      for (const { key, size } of all) {
        if (!key.startsWith(PREFIX_METADATA)) {
          result.push({
            name: key,
            size,
            metadata: metadataMap[key] || {
              originalSize: size,
              originalURL: "",
              etag: ""
            }
          });
        }
      }
      return result;
    });
  }
  /**
   * Clear all files currently in cache
   */
  clear() {
    return __async(this, null, function* () {
      yield this.deleteMany(() => true);
    });
  }
  /**
   * Delete a single file in cache
   *
   * @param nameOrURL Can be either an URL or a name returned by `getNameFromURL()` or `list()`
   */
  delete(nameOrURL) {
    return __async(this, null, function* () {
      const name2 = yield this.getNameFromURL(nameOrURL);
      yield this.deleteMany(
        (entry) => entry.name === nameOrURL || entry.name === name2
      );
    });
  }
  /**
   * Delete multiple files in cache.
   *
   * @param predicate A predicate like `array.filter(item => boolean)`
   */
  deleteMany(predicate) {
    return __async(this, null, function* () {
      const list = yield this.list();
      for (const item of list) {
        if (predicate(item)) {
          yield this.sb.delete(item.name);
          yield this.sb.delete(`${PREFIX_METADATA}${item.name}`);
        }
      }
    });
  }
  /**
   * Write the metadata of the file to disk.
   */
  writeMetadata(name, metadata) {
    return __async(this, null, function* () {
      const blob = new Blob([JSON.stringify(metadata)], { type: "text/plain" });
      yield this.sb.write(`${PREFIX_METADATA}${name}`, blob.stream());
    });
  }
};
var cache_manager_default = CacheManager;
function urlToFileName(url, prefix) {
  return __async(this, null, function* () {
    const hashBuffer = yield crypto.subtle.digest(
      "SHA-1",
      new TextEncoder().encode(url)
    );
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
    return `${prefix}${hashHex}_${url.split("/").pop()}`;
  });
}

// src/model-manager.ts
var DEFAULT_PARALLEL_DOWNLOADS = 3;
var ModelValidationStatus = /* @__PURE__ */ ((ModelValidationStatus2) => {
  ModelValidationStatus2["VALID"] = "valid";
  ModelValidationStatus2["INVALID"] = "invalid";
  ModelValidationStatus2["DELETED"] = "deleted";
  return ModelValidationStatus2;
})(ModelValidationStatus || {});
var Model = class {
  constructor(modelManager, url, mmprojUrl, savedFiles) {
    __publicField(this, "modelManager");
    /**
     * URL to the GGUF file (in case it contains multiple shards, the URL should point to the first shard)
     *
     * This URL will be used to identify the model in the cache. There can't be 2 models with the same URL.
     */
    __publicField(this, "url");
    /**
     * URL to mmproj file, if exists
     */
    __publicField(this, "mmprojUrl");
    /**
     * Size in bytes (total size of all shards).
     *
     * A value of -1 means the model is deleted from the cache. You must call `ModelManager.downloadModel` to re-download the model.
     */
    __publicField(this, "size");
    /**
     * List of all shards in the cache, sorted by original URL (ascending order)
     */
    __publicField(this, "files");
    this.modelManager = modelManager;
    this.url = url;
    this.mmprojUrl = mmprojUrl;
    if (savedFiles) {
      this.files = this.getAllFiles(savedFiles);
      this.size = sumArr(this.files.map((f) => f.metadata.originalSize));
    } else {
      this.files = [];
      this.size = 0;
    }
  }
  /**
   * Open and get a list of all shards as Blobs
   */
  open() {
    return __async(this, null, function* () {
      if (this.size === -1) {
        throw new WllamaError(
          `Model is deleted from the cache; Call ModelManager.downloadModel to re-download the model`,
          "load_error"
        );
      }
      const blobs = [];
      for (const file of this.files) {
        const blob = yield this.modelManager.cacheManager.open(file.name);
        if (!blob) {
          throw new Error(
            `Failed to open file ${file.name}; Hint: the model may be invalid, please refresh it`
          );
        }
        blobs.push(blob);
      }
      return blobs;
    });
  }
  /**
   * Validate the model files.
   *
   * If the model is invalid, the model manager will not be able to use it. You must call `refresh` to re-download the model.
   *
   * Cases that model is invalid:
   * - The model is deleted from the cache
   * - The model files are missing (or the download is interrupted)
   */
  validate() {
    let nbShards = ModelManager.parseModelUrl(this.url).length;
    if (this.mmprojUrl) {
      nbShards += 1;
    }
    if (this.size === -1) {
      return "deleted" /* DELETED */;
    }
    if (this.size < 16 || this.files.length !== nbShards) {
      return "invalid" /* INVALID */;
    }
    for (const file of this.files) {
      if (!file.metadata || file.metadata.originalSize !== file.size) {
        return "invalid" /* INVALID */;
      }
    }
    return "valid" /* VALID */;
  }
  /**
   * In case the model is invalid, call this function to re-download the model
   */
  refresh() {
    return __async(this, arguments, function* (options = {}) {
      var _a;
      const urls = ModelManager.parseModelUrl(this.url);
      if (this.mmprojUrl) {
        urls.push(this.mmprojUrl);
      }
      const works = urls.map((url, index) => ({
        url,
        index
      }));
      this.modelManager.logger.debug("Downloading model files:", urls);
      const nParallel = (_a = this.modelManager.params.parallelDownloads) != null ? _a : DEFAULT_PARALLEL_DOWNLOADS;
      const totalSize = yield this.getTotalDownloadSize(urls);
      const loadedSize = [];
      const worker = () => __async(this, null, function* () {
        while (works.length > 0) {
          const w = works.shift();
          if (!w) break;
          yield this.modelManager.cacheManager.download(w.url, __spreadProps(__spreadValues({}, options), {
            metadataAdditional: {
              originalURL: w.url,
              mmprojURL: this.mmprojUrl
            },
            progressCallback: ({ loaded }) => {
              var _a2;
              loadedSize[w.index] = loaded;
              (_a2 = options.progressCallback) == null ? void 0 : _a2.call(options, {
                loaded: sumArr(loadedSize),
                total: totalSize
              });
            }
          }));
        }
      });
      const promises = [];
      for (let i = 0; i < nParallel; i++) {
        promises.push(worker());
        loadedSize.push(0);
      }
      yield Promise.all(promises);
      this.files = this.getAllFiles(yield this.modelManager.cacheManager.list());
      this.size = this.files.reduce((acc, f) => acc + f.metadata.originalSize, 0);
    });
  }
  /**
   * Remove the model from the cache
   */
  remove() {
    return __async(this, null, function* () {
      this.files = this.getAllFiles(yield this.modelManager.cacheManager.list());
      yield this.modelManager.cacheManager.deleteMany(
        (f) => !!this.files.find((file) => file.name === f.name)
      );
      this.size = -1;
    });
  }
  getAllFiles(savedFiles) {
    const allUrls = new Set(ModelManager.parseModelUrl(this.url));
    if (this.mmprojUrl) {
      allUrls.add(this.mmprojUrl);
    }
    const allFiles = [];
    for (const url of allUrls) {
      const file = savedFiles.find((f) => f.metadata.originalURL === url);
      if (!file) {
        throw new Error(`Model file not found: ${url}`);
      }
      allFiles.push(file);
    }
    allFiles.sort(
      (a, b) => a.metadata.originalURL.localeCompare(b.metadata.originalURL)
    );
    return allFiles;
  }
  getTotalDownloadSize(urls) {
    return __async(this, null, function* () {
      const responses = yield Promise.all(
        urls.map((url) => fetch(url, { method: "HEAD" }))
      );
      const sizes = responses.map(
        (res) => Number(res.headers.get("content-length") || "0")
      );
      return sumArr(sizes);
    });
  }
};
var ModelManager = class _ModelManager {
  constructor(params = {}) {
    // The CacheManager singleton, can be accessed by user
    __publicField(this, "cacheManager");
    __publicField(this, "params");
    __publicField(this, "logger");
    this.cacheManager = params.cacheManager || new cache_manager_default();
    this.params = params;
    this.logger = params.logger || console;
  }
  /**
   * Parses a model URL and returns an array of URLs based on the following patterns:
   * - If the input URL is an array, it returns the array itself.
   * - If the input URL is a string in the `gguf-split` format, it returns an array containing the URL of each shard in ascending order.
   * - Otherwise, it returns an array containing the input URL as a single element array.
   * @param modelUrl URL or list of URLs
   */
  static parseModelUrl(modelUrl) {
    var _a;
    if (Array.isArray(modelUrl)) {
      return modelUrl;
    }
    const urlPartsRegex = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;
    const queryMatch = modelUrl.match(/\.gguf(\?.*)?$/);
    const queryParams = (_a = queryMatch == null ? void 0 : queryMatch[1]) != null ? _a : "";
    const matches = modelUrl.match(urlPartsRegex);
    if (!matches) {
      return [modelUrl];
    }
    const baseURL = modelUrl.replace(urlPartsRegex, "");
    const total = matches[2];
    const paddedShardIds = Array.from(
      { length: Number(total) },
      (_, index) => (index + 1).toString().padStart(5, "0")
    );
    return paddedShardIds.map(
      (current) => `${baseURL}-${current}-of-${total}.gguf${queryParams}`
    );
  }
  /**
   * Get all models in the cache
   */
  getModels() {
    return __async(this, arguments, function* (opts = {}) {
      const cachedFiles = yield this.cacheManager.list();
      let models = [];
      for (const file of cachedFiles) {
        if (!file.metadata.originalURL) continue;
        const shards = _ModelManager.parseModelUrl(file.metadata.originalURL);
        const mmprojUrl = file.metadata.mmprojURL;
        const isFirstShard = shards.length === 1 || shards[0] === file.metadata.originalURL;
        if (isFirstShard) {
          models.push(
            new Model(this, file.metadata.originalURL, mmprojUrl, cachedFiles)
          );
        }
      }
      if (!opts.includeInvalid) {
        models = models.filter(
          (m) => m.validate() === "valid" /* VALID */
        );
      }
      return models;
    });
  }
  /**
   * Download a model from the given URL.
   *
   * The URL must end with `.gguf`
   */
  downloadModel(_0) {
    return __async(this, arguments, function* (sourceOrURL, options = {}) {
      const source = isString(sourceOrURL) ? { url: sourceOrURL } : sourceOrURL;
      if (!isValidGgufFile(source.url)) {
        throw new WllamaError(
          `Invalid model URL: ${source.url}; URL must ends with ".gguf"`,
          "download_error"
        );
      }
      const model = new Model(this, source.url, source.mmprojUrl);
      const validity = model.validate();
      if (validity !== "valid" /* VALID */) {
        yield model.refresh(options);
      }
      return model;
    });
  }
  /**
   * Get a model from the cache or download it if it's not available.
   */
  getModelOrDownload(_0) {
    return __async(this, arguments, function* (source, options = {}) {
      var _a;
      const models = yield this.getModels();
      const model = models.find((m) => m.url === source.url);
      if (model) {
        (_a = options.progressCallback) == null ? void 0 : _a.call(options, { loaded: model.size, total: model.size });
        return model;
      }
      return this.downloadModel(source, options);
    });
  }
  /**
   * Remove all models from the cache
   */
  clear() {
    return __async(this, null, function* () {
      yield this.cacheManager.clear();
    });
  }
};

// src/types/types.ts
var LogLevel = /* @__PURE__ */ ((LogLevel2) => {
  LogLevel2[LogLevel2["DEBUG"] = 1] = "DEBUG";
  LogLevel2[LogLevel2["INFO"] = 2] = "INFO";
  LogLevel2[LogLevel2["WARN"] = 3] = "WARN";
  LogLevel2[LogLevel2["ERROR"] = 4] = "ERROR";
  return LogLevel2;
})(LogLevel || {});

// src/wasm-from-cdn.ts
var WasmCompatFromCDN = {
  worker: "https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@3.12.0/wasm/wllama.js",
  wasm: "https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@3.12.0/wasm/wllama.wasm"
};

// src/wllama.ts
var LoggerWithoutDebug = __spreadProps(__spreadValues({}, console), {
  debug: () => {
  }
});
var WllamaError = class extends Error {
  constructor(message, type = "unknown_error") {
    super(message);
    __publicField(this, "type");
    this.type = type;
  }
};
var WllamaAbortError = class extends Error {
  constructor() {
    super("Operation aborted");
    __publicField(this, "name", "AbortError");
  }
};
var WllamaRuntimeError = class extends Error {
  constructor(message, stack) {
    super(message);
    __publicField(this, "name", "RuntimeError");
    __publicField(this, "stack");
    this.stack = stack;
  }
};
var Wllama = class {
  constructor(pathConfig, wllamaConfig = {}) {
    // The CacheManager and ModelManager are singleton, can be accessed by user
    __publicField(this, "cacheManager");
    __publicField(this, "modelManager");
    /** Feature detection: `loraBlobs` means `lora_adapters` accepts GGUF blobs, `nativeStates` means `native_states` loads natlang native state spans, `natlangScore` means `createNatlangScore` is available, `slotState` means `saveSlot` / `restoreSlot` / `eraseSlot` / `applyTemplate` are available and completions accept `id_slot`, `nativeStateStore` means `putNativeState` / `getNativeState` are available. */
    __publicField(this, "capabilities", {
      loraBlobs: true,
      nativeStates: true,
      natlangScore: true,
      slotState: true,
      nativeStateStore: true
    });
    __publicField(this, "slotFileId", 0);
    __publicField(this, "compat", null);
    __publicField(this, "proxy", null);
    __publicField(this, "config");
    __publicField(this, "pathConfig");
    __publicField(this, "useMultiThread", false);
    __publicField(this, "nbThreads", 1);
    __publicField(this, "useEmbeddings", false);
    __publicField(this, "useRerank", false);
    // available when loaded
    __publicField(this, "loadedContextInfo", null);
    __publicField(this, "seed");
    __publicField(this, "bosToken", -1);
    __publicField(this, "eosToken", -1);
    __publicField(this, "eotToken", -1);
    __publicField(this, "eogTokens", /* @__PURE__ */ new Set());
    __publicField(this, "addBosToken", false);
    __publicField(this, "addEosToken", false);
    __publicField(this, "mediaMarker");
    __publicField(this, "chatTemplate");
    __publicField(this, "metadata");
    __publicField(this, "hasEncoder", false);
    __publicField(this, "decoderStartToken", -1);
    // note: we overlay instead of using llama-server default_template_kwargs, because we cannot transfer complex data structure via GLUE
    // overlay allow mixed data type or nested structure for kwargs
    __publicField(this, "chatTemplateKwargs", {});
    var _a, _b, _c;
    checkEnvironmentCompatible();
    if (!pathConfig) throw new WllamaError("AssetsPathConfig is required");
    this.pathConfig = pathConfig;
    this.config = wllamaConfig;
    this.cacheManager = (_a = wllamaConfig.cacheManager) != null ? _a : new cache_manager_default();
    this.modelManager = (_c = wllamaConfig.modelManager) != null ? _c : new ModelManager({
      cacheManager: this.cacheManager,
      logger: (_b = wllamaConfig.logger) != null ? _b : console,
      parallelDownloads: wllamaConfig.parallelDownloads,
      allowOffline: wllamaConfig.allowOffline
    });
    this.setCompat("default");
  }
  logger() {
    var _a;
    return (_a = this.config.logger) != null ? _a : console;
  }
  checkModelLoaded() {
    if (!this.isModelLoaded()) {
      throw new WllamaError(
        "loadModel() is not yet called",
        "model_not_loaded"
      );
    }
  }
  /**
   * Get the libllama version string, e.g. "b6327-4d74393".
   *
   * @returns version string embedded at build time.
   */
  static getLibllamaVersion() {
    return LIBLLAMA_VERSION;
  }
  /**
   * Set compatibility options for Wllama.
   * @param compat Set to null to disable compatibility, or 'default' to use the default compat resources from CDN.
   * @param mode 'safari' by default; If set to 'firefox_safari', the compat mode will **also** be enabled on Firefox, which will significantly degrade the performance but allow using WebGPU on Firefox.
   */
  setCompat(compat, mode = "safari") {
    if (mode === "safari") {
      if (isFirefox()) {
        this.compat = null;
        return;
      }
    }
    this.compat = compat === "default" ? WasmCompatFromCDN : compat;
  }
  /**
   * Check if the model is loaded via `loadModel()`
   */
  isModelLoaded() {
    return !!this.proxy && !!this.metadata;
  }
  /**
   * Get token ID associated to BOS (begin of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getBOS() {
    return this.bosToken;
  }
  /**
   * Get token ID associated to EOS (end of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOS() {
    return this.eosToken;
  }
  /**
   * Get token ID associated to EOT (end of turn) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOT() {
    return this.eotToken;
  }
  /**
   * Check if a given token is end-of-generation token (e.g. EOS, EOT, etc.)
   *
   * @param token the token ID to be checked
   * @returns true if the token is EOS, EOT, or any other end-of-generation tokens
   */
  isTokenEOG(token) {
    return token === this.eosToken || token === this.eotToken || this.eogTokens.has(token);
  }
  /**
   * Get token ID associated to token used by decoder, to start generating output sequence(only usable for encoder-decoder architecture). In other words, encoder uses normal BOS and decoder uses this token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getDecoderStartToken() {
    return this.decoderStartToken;
  }
  /**
   * Get model hyper-parameters and metadata
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns ModelMetadata
   */
  getModelMetadata() {
    this.checkModelLoaded();
    return this.metadata;
  }
  /**
   * Check if we're currently using multi-thread build.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isMultithread() {
    this.checkModelLoaded();
    return this.useMultiThread;
  }
  /**
   * Get number of threads used in the current context.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns number of threads
   */
  getNumThreads() {
    this.checkModelLoaded();
    return this.useMultiThread ? this.nbThreads : 1;
  }
  /**
   * Check if the current model uses encoder-decoder architecture
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isEncoderDecoderArchitecture() {
    this.checkModelLoaded();
    return this.hasEncoder;
  }
  /**
   * Must we add BOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if BOS token must be added to the sequence
   */
  mustAddBosToken() {
    this.checkModelLoaded();
    return this.addBosToken;
  }
  /**
   * Must we add EOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if EOS token must be added to the sequence
   */
  mustAddEosToken() {
    this.checkModelLoaded();
    return this.addEosToken;
  }
  /**
   * Get the jinja chat template comes with the model. It only available if the original model (before converting to gguf) has the template in `tokenizer_config.json`
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns the jinja template. null if there is no template in gguf
   */
  getChatTemplate() {
    var _a;
    this.checkModelLoaded();
    return (_a = this.chatTemplate) != null ? _a : null;
  }
  /**
   * Check if WebGPU is supported by the current environment.
   * @returns true if WebGPU is supported
   */
  isSupportWebGPU() {
    return isSupportWebGPU();
  }
  /**
   * Load model from a given URL (or a list of URLs, in case the model is splitted into smaller files)
   * - If the model already been downloaded (via `downloadModel()`), then we will use the cached model
   * - Else, we download the model from internet
   * @param modelSourceOrURL
   * @param params
   */
  loadModelFromUrl(_0) {
    return __async(this, arguments, function* (modelSourceOrURL, params = {}) {
      var _a;
      const source = isString(modelSourceOrURL) ? { url: modelSourceOrURL } : modelSourceOrURL;
      const useCache = (_a = params.useCache) != null ? _a : true;
      const model = useCache ? yield this.modelManager.getModelOrDownload(source, params) : yield this.modelManager.downloadModel(source, params);
      const blobs = yield model.open();
      return yield this.loadModel(blobs, params);
    });
  }
  /**
   * Load model from a given Hugging Face model ID and file path.
   *
   * @param hfOptions
   * @param params
   */
  loadModelFromHF(_0) {
    return __async(this, arguments, function* (hfOptions, params = {}) {
      const source = yield getHFModelSource(hfOptions);
      return yield this.loadModelFromUrl(source, params);
    });
  }
  /**
   * Load model from a given list of Blob.
   *
   * You can pass multiple buffers into the function (in case the model contains multiple shards).
   *
   * @param ggufBlobsOrModel Can be either list of Blobs (in case you use local file), or a Model object (in case you use ModelManager)
   * @param params LoadModelParams
   */
  loadModel(_0) {
    return __async(this, arguments, function* (ggufBlobsOrModel, params = {}) {
      var _a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k;
      const blobs = ggufBlobsOrModel instanceof Model ? yield ggufBlobsOrModel.open() : [...ggufBlobsOrModel];
      if (blobs.some((b) => b.size === 0)) {
        throw new WllamaError(
          "Input model (or splits) must be non-empty Blob or File",
          "load_error"
        );
      }
      if (!this.pathConfig["default"]) {
        throw new WllamaError(
          '"default" is missing from pathConfig',
          "load_error"
        );
      }
      if (this.proxy) {
        throw new WllamaError("Module is already initialized", "load_error");
      }
      const supportMultiThread = yield isSupportMultiThread();
      const hwConccurency = Math.floor((navigator.hardwareConcurrency || 1) / 2);
      const nbThreads = (_a = params.n_threads) != null ? _a : hwConccurency;
      this.nbThreads = nbThreads;
      this.useMultiThread = supportMultiThread && nbThreads > 1;
      const workerResources = this.getWorkerResources();
      if (params.n_gpu_layers === 0) {
        workerResources.noWebGPU = true;
      }
      this.proxy = new ProxyToWorker(
        workerResources,
        this.useMultiThread ? nbThreads : 0,
        // 0 means disable pthread
        (_b = this.config.suppressNativeLog) != null ? _b : false,
        this.logger()
      );
      let logLevel = (_c = params.log_level) != null ? _c : 2 /* INFO */;
      if (this.config.suppressNativeLog) {
        logLevel = 9999;
      }
      const modelFiles = yield prepareBlobs(blobs);
      const loraPaths = [];
      const sideFiles = [];
      for (const [i, adapter] of ((_d = params.lora_adapters) != null ? _d : []).entries()) {
        if (adapter.blob) {
          if (adapter.blob.size === 0) {
            throw new WllamaError(
              `LoRA adapter ${i} is an empty Blob`,
              "load_error"
            );
          }
          const name = `lora-${padDigits(i + 1, 5)}.gguf`;
          sideFiles.push({ name, blob: adapter.blob });
          loraPaths.push(`models/${name}`);
        } else if (adapter.path) {
          loraPaths.push(adapter.path);
        } else {
          throw new WllamaError(
            `LoRA adapter ${i} needs a path or a blob`,
            "load_error"
          );
        }
      }
      const statePaths = [];
      for (const [i, state] of ((_e = params.native_states) != null ? _e : []).entries()) {
        if (state.blob) {
          if (state.blob.size === 0) {
            throw new WllamaError(
              `Native state ${i} is an empty Blob`,
              "load_error"
            );
          }
          const name = `native-state-${padDigits(i + 1, 5)}.gguf`;
          sideFiles.push({ name, blob: state.blob });
          statePaths.push(`models/${name}`);
        } else if (state.path) {
          statePaths.push(state.path);
        } else {
          throw new WllamaError(
            `Native state ${i} needs a path or a blob`,
            "load_error"
          );
        }
      }
      yield this.proxy.moduleInit([...modelFiles.all, ...sideFiles]);
      this.logger().debug("Calling wllamaStart...");
      const startResult = yield this.proxy.wllamaStart();
      if (!startResult.success) {
        throw new WllamaError(
          `Error while calling start function, result = ${startResult}`
        );
      }
      this.logger().debug("Loading model...");
      const loadResult = yield this.proxy.wllamaAction("load", {
        _name: "load_req",
        log_level: logLevel,
        // if async read is not supported, use mmap; refer to README-dev.md for more details
        use_mmap: !canUseAsyncFileRead(workerResources.compat),
        use_mlock: false,
        n_gpu_layers: (_f = params.n_gpu_layers) != null ? _f : 99999,
        n_ctx: (_g = params.n_ctx) != null ? _g : 1024,
        n_threads: this.useMultiThread ? nbThreads : 1,
        n_ctx_auto: false,
        // not supported for now
        mmproj_path: modelFiles.mmproj ? `/models/${MMPROJ_FILE_NAME}` : void 0,
        model_paths: modelFiles.llm.map((f) => `models/${f.name}`),
        embeddings: params.embeddings,
        offload_kqv: params.offload_kqv,
        n_batch: params.n_batch,
        n_ubatch: params.n_ubatch,
        pooling_type: params.pooling_type,
        rope_scaling_type: params.rope_scaling_type,
        rope_freq_base: params.rope_freq_base,
        rope_freq_scale: params.rope_freq_scale,
        yarn_ext_factor: params.yarn_ext_factor,
        yarn_attn_factor: params.yarn_attn_factor,
        yarn_beta_fast: params.yarn_beta_fast,
        yarn_beta_slow: params.yarn_beta_slow,
        yarn_orig_ctx: params.yarn_orig_ctx,
        cache_type_k: params.cache_type_k,
        cache_type_v: params.cache_type_v,
        // with unified KV, all sequences share one n_ctx cache, so each request can still use the full context
        n_parallel: (_h = params.n_parallel) != null ? _h : 4,
        kv_unified: (_i = params.kv_unified) != null ? _i : true,
        flash_attn: params.flash_attn,
        swa_full: params.swa_full,
        chat_template: params.chat_template,
        jinja: params.jinja,
        reasoning: params.reasoning,
        image_min_tokens: params.image_min_tokens,
        image_max_tokens: params.image_max_tokens,
        warmup: params.warmup,
        no_kv_offload: params.no_kv_offload,
        mmproj_offload: params.mmproj_offload,
        cont_batching: params.cont_batching,
        n_keep: params.n_keep,
        ctx_shift: params.ctx_shift,
        cache_idle_slots: params.cache_idle_slots,
        n_cache_reuse: params.n_cache_reuse,
        lora_paths: params.lora_adapters ? loraPaths : void 0,
        lora_scales: (_j = params.lora_adapters) == null ? void 0 : _j.map((a) => {
          var _a2;
          return (_a2 = a.scale) != null ? _a2 : 1;
        }),
        lora_init_without_apply: params.lora_init_without_apply,
        native_state_paths: params.native_states ? statePaths : void 0,
        spec_draft_model: params.spec_draft_model,
        spec_draft_ngl: params.spec_draft_ngl,
        spec_draft_n_max: params.spec_draft_n_max,
        spec_draft_n_min: params.spec_draft_n_min,
        spec_draft_p_min: params.spec_draft_p_min,
        spec_draft_threads: params.spec_draft_threads,
        spec_draft_threads_batch: params.spec_draft_threads_batch,
        kv_overrides_keys: params.kv_overrides ? Object.keys(params.kv_overrides) : void 0,
        kv_overrides_vals: params.kv_overrides ? Object.values(params.kv_overrides) : void 0,
        reasoning_budget_tokens: params.reasoning_budget_tokens,
        reasoning_budget_message: params.reasoning_budget_message,
        reasoning_format: params.reasoning_format,
        skip_chat_parsing: params.skip_chat_parsing,
        prefill_assistant: params.prefill_assistant
      });
      const loadedCtxInfo = __spreadProps(__spreadValues({}, loadResult), {
        metadata: {}
      });
      for (let i = 0; i < loadResult.metadata_key.length; i++) {
        loadedCtxInfo.metadata[loadResult.metadata_key[i]] = loadResult.metadata_val[i];
      }
      this.seed = params.seed;
      this.bosToken = loadedCtxInfo.token_bos;
      this.eosToken = loadedCtxInfo.token_eos;
      this.eotToken = loadedCtxInfo.token_eot;
      this.useEmbeddings = !!params.embeddings;
      this.useRerank = params.pooling_type == "rank";
      this.metadata = {
        hparams: {
          nVocab: loadedCtxInfo.n_vocab,
          nCtxTrain: loadedCtxInfo.n_ctx_train,
          nEmbd: loadedCtxInfo.n_embd,
          nLayer: loadedCtxInfo.n_layer
        },
        meta: loadedCtxInfo.metadata
      };
      this.hasEncoder = !!loadedCtxInfo.has_encoder;
      this.decoderStartToken = loadedCtxInfo.token_decoder_start;
      this.addBosToken = loadedCtxInfo.add_bos_token;
      this.addEosToken = loadedCtxInfo.add_eos_token;
      this.chatTemplate = loadedCtxInfo.metadata["tokenizer.chat_template"];
      this.loadedContextInfo = loadedCtxInfo;
      this.eogTokens = new Set(loadedCtxInfo.list_tokens_eog);
      this.mediaMarker = loadedCtxInfo.media_marker;
      this.chatTemplateKwargs = (_k = params.default_template_kwargs) != null ? _k : {};
      this.logger().debug({ loadedCtxInfo });
    });
  }
  getLoadedContextInfo() {
    this.checkModelLoaded();
    if (!this.loadedContextInfo) {
      throw new WllamaError("Loaded context info is not available");
    }
    return __spreadValues({}, this.loadedContextInfo);
  }
  //////////////////////////////////////////////
  // High level API
  /**
   * Calculate embedding vector for a given text.
   * By default, BOS and EOS tokens will be added automatically. You can use the "skipBOS" and "skipEOS" option to disable it.
   * @param options OAI-compatible embedding creation options
   * @returns OAI-compatible embedding response
   */
  createEmbedding(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      if (!this.useEmbeddings) {
        throw new WllamaError(
          "Embeddings is not enabled. Please set it via LoadModelParams.embeddings"
        );
      }
      const result = yield this.proxy.wllamaAction(
        "embedding",
        {
          _name: "embd_req",
          data_json: JSON.stringify(options),
          files: []
          // TODO: support file input
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start inference",
          "inference_error"
        );
      }
      return yield this.getResponse(options, false, result.req_id);
    });
  }
  /**
   * Rerank a list of documents against a query.
   * Requires the model to be loaded with embeddings: true and pooling_type: 'rank'.
   * @param options Reranking options (query, documents, top_n)
   * @returns Reranking response with relevance scores sorted highest first
   */
  createRerank(options) {
    return __async(this, null, function* () {
      var _a, _b;
      this.checkModelLoaded();
      if (!this.useEmbeddings || !this.useRerank) {
        throw new WllamaError(
          "Rerank is not enabled. Please set it via LoadModelParams: embeddings = true and pooling_type = rank"
        );
      }
      const top_n = (_a = options.top_n) != null ? _a : options.documents.length;
      let totalTokens = 0;
      const rawResults = [];
      for (let i = 0; i < options.documents.length; i++) {
        const result = yield this.proxy.wllamaAction("rerank", {
          _name: "rrnk_req",
          data_json: JSON.stringify({
            query: options.query,
            document: options.documents[i]
          })
        });
        if (!result.success) {
          throw new WllamaError(
            "Model failed to start reranking",
            "inference_error"
          );
        }
        const { score, tokens_evaluated } = yield this.getSingleResult(result.req_id, "reranking");
        totalTokens += tokens_evaluated;
        rawResults.push({ index: i, score });
      }
      rawResults.sort((a, b) => b.score - a.score);
      return {
        model: (_b = this.getModelMetadata().meta["general.name"]) != null ? _b : "",
        object: "list",
        usage: { prompt_tokens: totalTokens, total_tokens: totalTokens },
        results: rawResults.slice(0, top_n).map(({ index, score }) => ({
          index,
          relevance_score: score
        }))
      };
    });
  }
  /**
   * Answer typed questions about a state, using a decision model (TypeSafe-compatible System One API).
   * Image input is not supported yet.
   * @param options The state and the questions
   * @returns The answer of each question
   */
  createSystemOne(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction(
        "systemone",
        {
          _name: "sys1_req",
          data_json: JSON.stringify(options)
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start systemone",
          "inference_error"
        );
      }
      return yield this.getSingleResult(
        result.req_id,
        "systemone"
      );
    });
  }
  /**
   * Score candidate replies to chat messages (natlang decision readout): the log-probability of each reply as the closed final assistant message, summed over its tokens after the chat prompt. The replies of an item share the prompt.
   * @param options The items to score
   * @returns Per item, the log-probability and token count of each reply, or the error of an item that could not be rendered
   */
  createNatlangScore(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction("score", {
        _name: "scor_req",
        data_json: JSON.stringify(options)
      });
      if (!result.success) {
        throw new WllamaError("Model failed to start scoring", "inference_error");
      }
      return yield this.getSingleResult(
        result.req_id,
        "score"
      );
    });
  }
  /**
   * Render chat messages (and tools) with the chat template, as the server's POST /apply-template does.
   * @param options OAI-compatible chat completion options (messages, tools, add_generation_prompt, ...)
   * @returns The prompt text
   */
  applyTemplate(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction(
        "apply_template",
        { _name: "tmpl_req", data_json: JSON.stringify(options) }
      );
      if (!result.success) {
        throw new WllamaError("Failed to apply chat template", "inference_error");
      }
      return result.prompt;
    });
  }
  /**
   * Save the state of a slot (its KV cache, tokens and context checkpoints), as the server's POST /slots/:id?action=save does. The slot must be idle.
   * @param idSlot The slot
   * @returns The saved state, to give to `restoreSlot` of an engine with the same model, build and context settings
   */
  saveSlot(idSlot) {
    return __async(this, null, function* () {
      const filename = `wllama-slot-${++this.slotFileId}.bin`;
      const res = yield this.slotAction(
        idSlot,
        "save",
        filename
      );
      const state = yield this.proxy.slotFileRead(filename);
      return { state, n_saved: res.n_saved, n_written: res.n_written };
    });
  }
  /**
   * Restore a slot from a state of `saveSlot`, as the server's POST /slots/:id?action=restore does. A state that does not fit leaves the slot empty and throws.
   * @param idSlot The slot
   * @param state The saved state
   * @returns The number of restored tokens and bytes read
   */
  restoreSlot(idSlot, state) {
    return __async(this, null, function* () {
      const filename = `wllama-slot-${++this.slotFileId}.bin`;
      const data = state instanceof Uint8Array ? state : new Uint8Array(yield state.arrayBuffer());
      yield this.proxy.slotFileWrite(filename, data);
      try {
        return yield this.slotAction(idSlot, "restore", filename);
      } finally {
        yield this.proxy.slotFileRemove(filename);
      }
    });
  }
  /**
   * Clear the cached prompt of a slot, as the server's POST /slots/:id?action=erase does.
   * @param idSlot The slot
   */
  eraseSlot(idSlot) {
    return __async(this, null, function* () {
      return yield this.slotAction(idSlot, "erase", "");
    });
  }
  slotAction(idSlot, action, filename) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction("slots", {
        _name: "slot_req",
        id_slot: idSlot,
        action,
        filename
      });
      if (!result.success) {
        throw new WllamaError(`Failed to ${action} slot`, "inference_error");
      }
      return yield this.getSingleResult(result.req_id, `slot ${action}`);
    });
  }
  /**
   * Store a natlang native state (natlang's store-file format, natlang.native-state/1) after the model is loaded, as the
   * server's PUT /v1/neuralese/native-states/:id does: the bytes are verified (layout, finite values, content ID), kept as
   * received and, for a state of this model's identity, registered so prompts consume it exactly.
   * @param state The state's bytes
   * @param id The ID the state must have (its content ID); omit to accept any
   * @returns The consumption mode and the state's metadata ({id, ...metadata})
   */
  putNativeState(state, id = "") {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const filename = `wllama-native-state-${++this.slotFileId}.nzs`;
      const data = state instanceof Uint8Array ? state : new Uint8Array(yield state.arrayBuffer());
      yield this.proxy.slotFileWrite(filename, data);
      try {
        const result = yield this.proxy.wllamaAction(
          "native_state",
          { _name: "nsta_req", action: "put", id, filename }
        );
        if (!result.success) {
          throw new WllamaError("Failed to put native state", "inference_error");
        }
        return yield this.getSingleResult(result.req_id, "native state put");
      } finally {
        yield this.proxy.slotFileRemove(filename);
      }
    });
  }
  /**
   * Read a stored native state (one this engine produced, or one put or loaded), as the server's
   * GET /v1/neuralese/native-states/:id does: its bytes exactly as stored, natlang's store-file format.
   * @param id The state's ID
   * @returns The state's bytes and its metadata ({id, ...metadata})
   */
  getNativeState(id) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const filename = `wllama-native-state-${++this.slotFileId}.nzs`;
      const result = yield this.proxy.wllamaAction(
        "native_state",
        { _name: "nsta_req", action: "get", id, filename }
      );
      if (!result.success) {
        throw new WllamaError("Failed to get native state", "inference_error");
      }
      const state = yield this.proxy.slotFileRead(filename);
      return { state, meta: JSON.parse(result.meta_json) };
    });
  }
  createChatCompletion(options) {
    return __async(this, null, function* () {
      var _a;
      if (Object.keys(this.chatTemplateKwargs).length > 0) {
        options = __spreadProps(__spreadValues({}, options), {
          chat_template_kwargs: __spreadValues(__spreadValues({}, this.chatTemplateKwargs), (_a = options.chat_template_kwargs) != null ? _a : {})
        });
      }
      if (options.stream && options.onData) {
        yield this.createCompletionImpl(options);
      } else if (options.stream) {
        return yield this.createCompletionGenerator(options);
      } else {
        return yield this.createCompletionImpl(__spreadProps(__spreadValues({}, options), { stream: false }));
      }
    });
  }
  createCompletion(options) {
    return __async(this, null, function* () {
      if (options.stream && options.onData) {
        yield this.createCompletionImpl(options);
      } else if (options.stream) {
        return yield this.createCompletionGenerator(options);
      } else {
        return yield this.createCompletionImpl(__spreadProps(__spreadValues({}, options), { stream: false }));
      }
    });
  }
  /**
   * Private implementation of createCompletion
   */
  createCompletionImpl(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const isStream = !!options.stream;
      const isChat = !!options.messages;
      const customOpt = {};
      if (this.seed !== void 0) {
        customOpt.seed = this.seed;
      }
      let files = [];
      if (isChat) {
        const tmp = this.prepareMultimodalInput(
          options
        );
        options = tmp.params;
        files = tmp.files;
      }
      const result = yield this.proxy.wllamaAction(
        "completion",
        {
          _name: "cmpl_req",
          is_chat: isChat,
          data_json: JSON.stringify(__spreadValues(__spreadValues({}, options), customOpt)),
          files: files.map((f) => new Uint8Array(f))
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start inference",
          "inference_error"
        );
      }
      return yield this.getResponse(
        options,
        isStream,
        result.req_id
      );
    });
  }
  /**
   * Same with `createCompletion`, but returns an async iterator instead.
   * Only called when stream=true and no onData is provided.
   */
  createCompletionGenerator(options) {
    return new Promise((resolve) => {
      const createGenerator = cbToAsyncIter(
        (callback) => {
          this.createCompletionImpl(__spreadProps(__spreadValues({}, options), {
            onData: (chunk) => callback(chunk)
          })).then(() => callback(void 0, true)).catch((err) => callback(void 0, false, err));
        }
      );
      resolve(createGenerator());
    });
  }
  /**
   * Whether the currently loaded model supports a specific input modality (e.g. image or audio).
   * @param modality
   * @returns
   */
  supportInputModality(modality) {
    this.checkModelLoaded();
    if (modality === "image") {
      return !!this.loadedContextInfo.has_image_input;
    } else if (modality === "audio") {
      return !!this.loadedContextInfo.has_audio_input;
    } else {
      throw new WllamaError(
        "Unsupported modality: " + modality,
        "unknown_error"
      );
    }
  }
  /**
   * Unload the model and free all memory.
   *
   * Note: This function will NOT crash if model is not yet loaded
   */
  exit() {
    return __async(this, null, function* () {
      var _a;
      yield (_a = this.proxy) == null ? void 0 : _a.wllamaExit();
      this.proxy = null;
    });
  }
  /**
   * [FOR DEBUGGING ONLY] Run ggml backend ops tests without loading any model.
   *
   * Initializes the wasm runtime, executes `test-backend-ops` with the given args, then shuts down.
   *
   * For more info, please refer to guides/debug.md
   *
   * @param args Arguments forwarded to test-backend-ops (e.g. ["-o", "ADD"])
   * @returns retcode (0 = all tests passed) and success flag
   */
  testBackendOps() {
    return __async(this, arguments, function* (args = []) {
      var _a;
      if (!this.pathConfig["default"]) {
        throw new WllamaError(
          '"default" is missing from pathConfig',
          "load_error"
        );
      }
      if (!(yield isSupportMultiThread())) {
        throw new WllamaError(
          "Multi-threading is required to run backend ops tests, but it is not supported in the current environment."
        );
      }
      const tmpProxy = new ProxyToWorker(
        this.getWorkerResources(),
        0,
        // single-thread; no model needed
        (_a = this.config.suppressNativeLog) != null ? _a : false,
        this.logger()
      );
      try {
        yield tmpProxy.moduleInit([]);
        const startResult = yield tmpProxy.wllamaStart();
        if (!startResult.success) {
          throw new WllamaError(
            `Error while calling start function, result = ${startResult}`
          );
        }
        const result = yield tmpProxy.wllamaAction(
          "test_backend_ops",
          { _name: "tbop_req", args: ["test-backend-ops", ...args] }
        );
        return { retcode: result.retcode, success: result.success };
      } finally {
        yield tmpProxy.wllamaExit();
      }
    });
  }
  //////////////////////////////////////////////
  // Low level API
  // TODO: add back
  /**
   * get debug info
   */
  _getDebugInfo() {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      return yield this.proxy.wllamaDebug();
    });
  }
  //////////////////////////////////////////////
  // Utils
  jsonDecode(data_json) {
    try {
      return JSON.parse(data_json);
    } catch (e) {
      this.logger().error("Failed to parse JSON:", data_json);
      throw new WllamaError("Failed to parse model output", "inference_error");
    }
  }
  prepareMultimodalInput(params) {
    const msg = params.messages;
    const msgNew = [];
    const files = [];
    for (const m of msg) {
      if (Array.isArray(m.content)) {
        const newContent = [];
        for (const c of m.content) {
          if (c.type === "text" || c.type === "native_state") {
            newContent.push(c);
          } else {
            if (!this.mediaMarker) {
              throw new WllamaError(
                "Media marker is undefined",
                "inference_error"
              );
            }
            files.push(c.data);
            newContent.push({
              type: "text",
              text: this.mediaMarker
            });
          }
        }
        msgNew.push(__spreadProps(__spreadValues({}, m), {
          content: newContent
        }));
      } else {
        msgNew.push(m);
      }
    }
    return {
      params: __spreadProps(__spreadValues({}, params), {
        messages: msgNew
      }),
      files
    };
  }
  // release the slot occupied by the request; cancelling an already-finished request is a no-op
  cancelRequest(reqId) {
    return __async(this, null, function* () {
      try {
        yield this.proxy.wllamaAction("cancel", {
          _name: "cncl_req",
          req_id: reqId
        });
      } catch (e) {
        this.logger().warn("Failed to cancel request", reqId, e);
      }
    });
  }
  getSingleResult(reqId, task) {
    return __async(this, null, function* () {
      let completed = false;
      try {
        while (true) {
          const chunk = yield this.proxy.wllamaAction(
            "get_result",
            { _name: "gres_req", req_id: reqId }
          );
          const jsonString = chunk.data_json;
          if (jsonString && jsonString.length > 0) {
            if (chunk.is_error) {
              const jsonData = this.jsonDecode(jsonString);
              throw new WllamaError(
                jsonData.message || `Unknown ${task} error`,
                "inference_error"
              );
            }
            completed = true;
            return this.jsonDecode(jsonString);
          }
          if (!chunk.has_more) {
            completed = true;
            break;
          }
        }
        throw new WllamaError(`No ${task} result received`, "inference_error");
      } finally {
        if (!completed) {
          yield this.cancelRequest(reqId);
        }
      }
    });
  }
  getResponse(options, isStream, reqId) {
    return __async(this, null, function* () {
      var _a, _b;
      let finalResult = null;
      let completed = false;
      try {
        while (true) {
          if ((_a = options.abortSignal) == null ? void 0 : _a.aborted) {
            throw new WllamaAbortError();
          }
          const result_chunk = yield this.proxy.wllamaAction(
            "get_result",
            {
              _name: "gres_req",
              req_id: reqId
            }
          );
          const jsonString = result_chunk.data_json;
          if (!jsonString || jsonString.length === 0) {
            if (!result_chunk.has_more) {
              completed = true;
              break;
            } else {
              continue;
            }
          }
          if (jsonString == "null") {
            continue;
          }
          let jsonData = this.jsonDecode(jsonString);
          finalResult = jsonData;
          if (result_chunk.is_error) {
            this.logger().error("Model returned an error:", jsonData);
            throw new WllamaError(
              jsonData.message || "Unknown inference error",
              "inference_error"
            );
          }
          if (isStream) {
            if (!Array.isArray(jsonData)) {
              jsonData = [jsonData];
            }
            for (const chunk of jsonData) {
              (_b = options.onData) == null ? void 0 : _b.call(options, chunk);
              finalResult = chunk;
            }
          }
          if (!result_chunk.has_more) {
            completed = true;
            break;
          }
        }
      } finally {
        if (!completed) {
          yield this.cancelRequest(reqId);
        }
      }
      return finalResult;
    });
  }
  getWorkerResources() {
    const workerResources = {
      wasmPath: absoluteUrl(this.pathConfig["default"]),
      compat: false
    };
    if (needCompat()) {
      if (!this.compat) {
        this.logger().warn(
          "Not using compat mode" + (isFirefox() ? " (expected on Firefox - WebGPU will be disabled)" : "")
        );
      } else {
        const isUsingDefault = this.compat.worker === WasmCompatFromCDN.worker && this.compat.wasm === WasmCompatFromCDN.wasm;
        if (isUsingDefault) {
          this.logger().warn(
            "Compatibility mode is activated, using resources from CDN. To use local resources, please refer to @wllama/wllama-compat package."
          );
          this.logger().warn(
            "IMPORTANT: Performance will be significantly degraded in compatibility mode."
          );
        }
        workerResources.wasmPath = absoluteUrl(this.compat.wasm);
        workerResources.jsPath = this.compat.worker;
        workerResources.compat = true;
      }
    }
    if (isFirefox()) {
      if (workerResources.compat) {
        this.logger().warn(
          'Using compat mode on Firefox, performance will be significantly degraded; Consider enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      } else if (!isSupportJSPI()) {
        this.logger().warn(
          'WebGPU is disabled on Firefox due to missing JSPI support. Please consider enabling compat mode, or enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      }
    }
    return workerResources;
  }
};
export {
  CacheManager,
  LogLevel,
  LoggerWithoutDebug,
  Model,
  ModelManager,
  ModelValidationStatus,
  POLYFILL_ETAG,
  Wllama,
  WllamaAbortError,
  WllamaError,
  WllamaRuntimeError,
  getHFFileSHA256,
  getHFModelSource,
  isValidGgufFile
};
